from fuel.streams import AbstractDataStream

class RetinaStream(AbstractDataStream):
    """
    A fuel DataStream for retina data

    from fuel:
    A data stream is an iterable stream of examples/minibatches. It shares
    similarities with Python file handles return by the ``open`` method.
    Data streams can be closed using the :meth:`close` method and reset
    using :meth:`reset` (similar to ``f.seek(0)``).


    """

    def __init__(self):
