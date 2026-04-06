from src.core.errors.exception_handler import AppException


def main():
    try:
        a = 1/0
    except Exception as e:
        raise AppException(e)
    

if __name__ == "__main__":
    main()