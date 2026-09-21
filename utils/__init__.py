from utils.cvfpscalc import CvFpsCalc

__all__ = ['CvFpsCalc', 'CvDrawText']


def __getattr__(name):
    # Keep Pillow optional for the simple demos and CSV readers.
    if name == 'CvDrawText':
        from utils.cvdrawtext import CvDrawText
        return CvDrawText
    raise AttributeError(f'module {__name__!r} has no attribute {name!r}')
