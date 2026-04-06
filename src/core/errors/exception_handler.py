from __future__ import annotations

import sys
from pathlib import Path
from traceback import TracebackException
from types import TracebackType
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class ExcInfoProvider(Protocol):
    """Объект, совместимый с интерфейсом sys.exc_info()."""

    def exc_info(
        self,
    ) -> tuple[
        Optional[type[BaseException]],
        Optional[BaseException],
        Optional[TracebackType],
    ]:
        ...


class AppException(Exception):
    """Production-ready обёртка над исходным исключением.

    Хранит исходную ошибку, тип ошибки, файл и строку, где произошло
    фактическое исключение, а также готовое человекочитаемое сообщение.

    Пример:
        try:
            ...
        except Exception as e:
            raise AppException(e) from e
    """

    __slots__ = (
        "original_error",
        "error_type",
        "file_name",
        "line_number",
        "error_message",
    )

    def __init__(
        self,
        error: BaseException,
        error_detail: ExcInfoProvider = sys,
    ) -> None:
        if not isinstance(error, BaseException):
            raise TypeError("error must be an instance of BaseException")

        self.original_error = error
        self.error_type = type(error).__name__

        traceback_obj = self._resolve_traceback(error, error_detail)
        self.file_name, self.line_number = self._extract_location(traceback_obj)
        self.error_message = self._build_message()

        super().__init__(self.error_message)

    @staticmethod
    def _resolve_traceback(
        error: BaseException,
        error_detail: ExcInfoProvider,
    ) -> Optional[TracebackType]:
        """Возвращает traceback из ошибки или из совместимого exc_info()."""
        if error.__traceback__ is not None:
            return error.__traceback__

        exc_info = getattr(error_detail, "exc_info", None)
        if not callable(exc_info):
            return None

        try:
            _, _, traceback_obj = exc_info()
        except Exception:
            return None

        return traceback_obj

    @staticmethod
    def _extract_location(
        traceback_obj: Optional[TracebackType],
    ) -> tuple[Optional[str], Optional[int]]:
        """Извлекает реальное место падения из последнего кадра traceback."""
        if traceback_obj is None:
            return None, None

        current = traceback_obj
        while current.tb_next is not None:
            current = current.tb_next

        return current.tb_frame.f_code.co_filename, current.tb_lineno

    def _build_message(self) -> str:
        base_message = f"{self.error_type}: {self.original_error}"

        if self.file_name is None or self.line_number is None:
            return f"Произошла ошибка. {base_message}"

        return (
            "Упс! Произошла ошибка в коде. "
            f"{self._display_file_name(self.file_name)}: "
            f"на строке {self.line_number}: {base_message}"
        )

    @staticmethod
    def _display_file_name(file_name: str) -> str:
        """Нормализует путь для более удобного отображения."""
        try:
            return str(Path(file_name))
        except Exception:
            return file_name

    def to_dict(self) -> dict[str, object]:
        """Возвращает сериализуемое представление для логирования."""
        return {
            "error_type": self.error_type,
            "message": str(self.original_error),
            "file_name": self.file_name,
            "line_number": self.line_number,
            "full_message": self.error_message,
        }

    def format_traceback(self) -> str:
        """Возвращает полный форматированный traceback исходной ошибки."""
        return "".join(
            TracebackException.from_exception(self.original_error).format()
        ).rstrip()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(error_type={self.error_type!r}, "
            f"file_name={self.file_name!r}, line_number={self.line_number!r}, "
            f"error_message={self.error_message!r})"
        )

    def __str__(self) -> str:
        return self.error_message