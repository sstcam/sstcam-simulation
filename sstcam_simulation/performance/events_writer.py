import tables


FILTERS = tables.Filters(
    complevel=5,  # compression medium, tradeoff between speed and compression
    complib="blosc:zstd",  # use modern zstd algorithm
    fletcher32=True,  # add checksums to data chunks
)


class EventsWriter:
    def __init__(self, path, events_table_layout):
        print(f"Creating file: {path}")
        self._file = tables.File(path, mode='w', filters=FILTERS)
        self._table = self._file.create_table(
            self._file.root, "event", events_table_layout, "Events"
        )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._file.close()

    def flush(self):
        self._file.flush()

    def add_metadata(self, **kwargs):
        for key, value in kwargs.items():
            self._table.attrs[key] = value

    def append(self, event):
        row = self._table.row
        for key, value in event.items():
            row[key] = value
        row.append()
