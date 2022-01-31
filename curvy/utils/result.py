import overpy


class QueryResult:
    def __init__(self, result: overpy.Result, railway_type: str):
        self.railway_type = railway_type
        self.result: overpy.Result = result

    @property
    def result(self):
        return self._result

    @result.setter
    def result(self, value: overpy.Result):
        # Assign ways to nodes
        for way in value.ways:
            for node in way.nodes:
                if hasattr(node, "ways"):
                    node.ways.append(way)
                else:
                    node.ways = [way]

        self._result = value