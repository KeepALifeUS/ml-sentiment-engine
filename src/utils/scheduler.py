"""
Task scheduling utilities для ML-Framework ML Sentiment Engine

Enterprise-grade планировщик задач с async support и Context7 patterns.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import uuid

from .logger import get_logger
from .config import get_config

logger = get_logger(__name__)


class TaskStatus(str, Enum):
    """Статусы задач"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskPriority(int, Enum):
    """Приоритеты задач"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class ScheduledTask:
    """Структура запланированной задачи"""
    
    id: str
    name: str
    function: Callable
    args: tuple = ()
    kwargs: dict = None
    priority: TaskPriority = TaskPriority.NORMAL
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[Exception] = None
    result: Any = None
    retry_count: int = 0
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: Optional[float] = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if self.created_at is None:
            self.created_at = datetime.utcnow()


class SentimentTaskScheduler:
    """
    Enterprise-grade асинхронный планировщик задач
    
    Features:
    - Приоритетная очередь задач
    - Retry логика с exponential backoff
    - Timeout handling
    - Concurrent execution control
    - Performance monitoring
    - Circuit breaker integration
    """
    
    def __init__(self, max_concurrent_tasks: Optional[int] = None):
        """
        Инициализация планировщика
        
        Args:
            max_concurrent_tasks: Максимальное количество одновременных задач
        """
        config = get_config()
        self.max_concurrent_tasks = max_concurrent_tasks or config.max_concurrent_tasks
        
        self.tasks: Dict[str, ScheduledTask] = {}
        self.pending_queue: List[ScheduledTask] = []
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.completed_tasks: List[ScheduledTask] = []
        
        self._running = False
        self._executor_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()
        
        # Metrics
        self.total_tasks = 0
        self.successful_tasks = 0
        self.failed_tasks = 0
        self.cancelled_tasks = 0
    
    async def start(self):
        """Запуск планировщика"""
        if self._running:
            logger.warning("Scheduler already running")
            return
        
        self._running = True
        self._executor_task = asyncio.create_task(self._executor_loop())
        logger.info("Task scheduler started", max_concurrent_tasks=self.max_concurrent_tasks)
    
    async def stop(self):
        """Остановка планировщика"""
        if not self._running:
            logger.warning("Scheduler not running")
            return
        
        self._running = False
        
        # Отмена всех выполняющихся задач
        for task_id, task in self.running_tasks.items():
            task.cancel()
            self.tasks[task_id].status = TaskStatus.CANCELLED
            self.cancelled_tasks += 1
        
        # Ожидание завершения executor loop
        if self._executor_task:
            self._executor_task.cancel()
            try:
                await self._executor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Task scheduler stopped")
    
    def schedule_task(
        self,
        name: str,
        function: Callable,
        args: tuple = (),
        kwargs: dict = None,
        priority: TaskPriority = TaskPriority.NORMAL,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        timeout: Optional[float] = None
    ) -> str:
        """
        Планирование задачи для выполнения
        
        Args:
            name: Название задачи
            function: Функция для выполнения
            args: Аргументы функции
            kwargs: Keyword аргументы функции
            priority: Приоритет задачи
            max_retries: Максимальное количество попыток
            retry_delay: Задержка между попытками
            timeout: Timeout выполнения
            
        Returns:
            str: ID созданной задачи
        """
        task_id = str(uuid.uuid4())
        
        task = ScheduledTask(
            id=task_id,
            name=name,
            function=function,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            max_retries=max_retries,
            retry_delay=retry_delay,
            timeout=timeout
        )
        
        self.tasks[task_id] = task
        self.pending_queue.append(task)
        self.total_tasks += 1
        
        # Сортировка очереди по приоритету
        self.pending_queue.sort(key=lambda t: t.priority.value, reverse=True)
        
        logger.debug(
            "Task scheduled",
            task_id=task_id,
            task_name=name,
            priority=priority.name,
            queue_size=len(self.pending_queue)
        )
        
        return task_id
    
    def schedule_recurring_task(
        self,
        name: str,
        function: Callable,
        interval: float,
        args: tuple = (),
        kwargs: dict = None,
        priority: TaskPriority = TaskPriority.NORMAL
    ) -> str:
        """
        Планирование повторяющейся задачи
        
        Args:
            name: Название задачи
            function: Функция для выполнения
            interval: Интервал повторения в секундах
            args: Аргументы функции
            kwargs: Keyword аргументы
            priority: Приоритет задачи
            
        Returns:
            str: ID созданной задачи
        """
        async def recurring_wrapper():
            """Wrapper для повторяющихся задач"""
            while self._running:
                try:
                    if asyncio.iscoroutinefunction(function):
                        await function(*args, **kwargs)
                    else:
                        function(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Error in recurring task {name}", error=e)
                
                await asyncio.sleep(interval)
        
        return self.schedule_task(
            name=f"{name}_recurring",
            function=recurring_wrapper,
            priority=priority,
            max_retries=0  # Recurring tasks don't retry
        )
    
    def cancel_task(self, task_id: str) -> bool:
        """
        Отмена задачи
        
        Args:
            task_id: ID задачи для отмены
            
        Returns:
            bool: True если задача была отменена
        """
        if task_id not in self.tasks:
            return False
        
        task = self.tasks[task_id]
        
        if task.status == TaskStatus.RUNNING:
            # Отмена выполняющейся задачи
            if task_id in self.running_tasks:
                self.running_tasks[task_id].cancel()
            task.status = TaskStatus.CANCELLED
            self.cancelled_tasks += 1
        elif task.status == TaskStatus.PENDING:
            # Удаление из очереди
            self.pending_queue = [t for t in self.pending_queue if t.id != task_id]
            task.status = TaskStatus.CANCELLED
            self.cancelled_tasks += 1
        
        logger.info(f"Task cancelled", task_id=task_id, task_name=task.name)
        return True
    
    def get_task_status(self, task_id: str) -> Optional[ScheduledTask]:
        """
        Получение статуса задачи
        
        Args:
            task_id: ID задачи
            
        Returns:
            Optional[ScheduledTask]: Информация о задаче или None
        """
        return self.tasks.get(task_id)
    
    def get_pending_tasks(self) -> List[ScheduledTask]:
        """Получение списка ожидающих задач"""
        return self.pending_queue.copy()
    
    def get_running_tasks(self) -> List[ScheduledTask]:
        """Получение списка выполняющихся задач"""
        return [self.tasks[task_id] for task_id in self.running_tasks.keys()]
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Получение статистики планировщика
        
        Returns:
            Dict[str, Any]: Статистика планировщика
        """
        return {
            "total_tasks": self.total_tasks,
            "successful_tasks": self.successful_tasks,
            "failed_tasks": self.failed_tasks,
            "cancelled_tasks": self.cancelled_tasks,
            "pending_tasks": len(self.pending_queue),
            "running_tasks": len(self.running_tasks),
            "completed_tasks": len(self.completed_tasks),
            "success_rate": self.successful_tasks / max(self.total_tasks, 1),
            "is_running": self._running
        }
    
    async def _executor_loop(self):
        """Основной цикл выполнения задач"""
        logger.info("Task executor loop started")
        
        while self._running:
            try:
                # Проверка доступности слотов для задач
                if len(self.running_tasks) >= self.max_concurrent_tasks:
                    await asyncio.sleep(0.1)
                    continue
                
                # Получение следующей задачи из очереди
                async with self._lock:
                    if not self.pending_queue:
                        await asyncio.sleep(0.1)
                        continue
                    
                    next_task = self.pending_queue.pop(0)
                
                # Запуск задачи
                await self._execute_task(next_task)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Error in executor loop", error=e)
                await asyncio.sleep(1.0)
        
        logger.info("Task executor loop stopped")
    
    async def _execute_task(self, task: ScheduledTask):
        """
        Выполнение одной задачи
        
        Args:
            task: Задача для выполнения
        """
        task_start = datetime.utcnow()
        task.status = TaskStatus.RUNNING
        task.started_at = task_start
        
        logger.info(
            "Task started",
            task_id=task.id,
            task_name=task.name,
            retry_count=task.retry_count
        )
        
        try:
            # Создание asyncio.Task для выполнения
            if asyncio.iscoroutinefunction(task.function):
                coro = task.function(*task.args, **task.kwargs)
            else:
                # Обертка для синхронных функций
                coro = self._run_sync_function(task.function, task.args, task.kwargs)
            
            # Выполнение с timeout
            if task.timeout:
                async_task = asyncio.wait_for(coro, timeout=task.timeout)
            else:
                async_task = coro
            
            # Добавление в running_tasks
            self.running_tasks[task.id] = asyncio.create_task(async_task)
            
            # Ожидание завершения
            task.result = await self.running_tasks[task.id]
            
            # Успешное завершение
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.utcnow()
            self.successful_tasks += 1
            self.completed_tasks.append(task)
            
            execution_time = (task.completed_at - task_start).total_seconds() * 1000
            logger.info(
                "Task completed successfully",
                task_id=task.id,
                task_name=task.name,
                execution_time_ms=execution_time
            )
            
        except asyncio.CancelledError:
            task.status = TaskStatus.CANCELLED
            self.cancelled_tasks += 1
            logger.info(f"Task cancelled", task_id=task.id, task_name=task.name)
            
        except Exception as e:
            task.error = e
            
            # Проверка возможности retry
            if task.retry_count < task.max_retries:
                task.retry_count += 1
                task.status = TaskStatus.PENDING
                
                # Exponential backoff
                delay = task.retry_delay * (2 ** (task.retry_count - 1))
                await asyncio.sleep(delay)
                
                # Возвращение в очередь
                async with self._lock:
                    self.pending_queue.append(task)
                    self.pending_queue.sort(key=lambda t: t.priority.value, reverse=True)
                
                logger.warning(
                    "Task failed, retrying",
                    task_id=task.id,
                    task_name=task.name,
                    retry_count=task.retry_count,
                    max_retries=task.max_retries,
                    delay=delay,
                    error=str(e)
                )
            else:
                # Превышено количество попыток
                task.status = TaskStatus.FAILED
                task.completed_at = datetime.utcnow()
                self.failed_tasks += 1
                
                logger.error(
                    "Task failed permanently",
                    task_id=task.id,
                    task_name=task.name,
                    retry_count=task.retry_count,
                    error=e
                )
        
        finally:
            # Удаление из running_tasks
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]
    
    async def _run_sync_function(self, function: Callable, args: tuple, kwargs: dict) -> Any:
        """
        Выполнение синхронной функции в asyncio executor
        
        Args:
            function: Синхронная функция
            args: Аргументы
            kwargs: Keyword аргументы
            
        Returns:
            Any: Результат выполнения функции
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: function(*args, **kwargs))


# Global scheduler instance
_scheduler: Optional[SentimentTaskScheduler] = None


def get_scheduler() -> SentimentTaskScheduler:
    """
    Получение singleton instance планировщика
    
    Returns:
        SentimentTaskScheduler: Планировщик задач
    """
    global _scheduler
    if _scheduler is None:
        _scheduler = SentimentTaskScheduler()
    return _scheduler


async def start_scheduler():
    """Запуск глобального планировщика"""
    scheduler = get_scheduler()
    await scheduler.start()


async def stop_scheduler():
    """Остановка глобального планировщика"""
    scheduler = get_scheduler()
    await scheduler.stop()