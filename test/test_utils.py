# -*- coding:utf8

from agilkia.utils import *

import unittest


class TestColor(unittest.TestCase):
    def test_average(self):
        color1 = Color(255, 0, 245)
        color2 = Color(45, 23, 0)
        self.assertIsNone(color1.alpha)
        self.assertIsNone(color2.alpha)

        average = color1.average(color2, 0.5)
        self.assertEqual(average.r, int((color1.r + color2.r) / 2))
        self.assertEqual(average.g, int((color1.g + color2.g) / 2))
        self.assertEqual(average.b, int((color1.b + color2.b) / 2))
        self.assertIsNone(average.alpha)

        color1.alpha = 34
        average = color1.average(color2, 0.5)
        self.assertIsNone(average.alpha)

        color2.alpha = 235
        average = color1.average(color2, 0.5)
        self.assertEqual(average.alpha, int((color1.alpha + color2.alpha) / 2))

        color1.alpha = None
        average = color1.average(color2, 0.5)
        self.assertIsNone(average.alpha)

    def test_toHex(self):
        color = Color(0x07, 0x68, 0xF4)
        self.assertIsNone(color.alpha)
        self.assertEqual(color.toHex(), '#0768f4')
        color.alpha = 0xe2
        self.assertEqual(color.toHex(), '#0768f4e2')


class TestColorList(unittest.TestCase):
    def test_exist_and_change(self):
        """
        Might fail with probability of 100(number of loops) / 2^24 (number of existing colors) if the color list randomly take two times the same color.
        """
        colorList = ColorList()
        prev = colorList.pickColor()
        self.assertIsNotNone(prev)
        for i in range(100):
            color = colorList.pickColor()
            self.assertIsNotNone(color)
            self.assertTrue(color.r != prev.r or color.g != prev.g or color.b != prev.b)
            prev = color


#class TestTraceLoader(unittest.TestCase):
#    mainTmpDir = tempfile.TemporaryDirectory()
#
#    def setUp(self):
#        self.tmpDir = os.path.join(self.mainTmpDir.name, "test cache")
#        TracesLoader.cacheDir = self.tmpDir
#
#    def removeCache(self):
#        if os.path.exists(self.tmpDir):
#            for f in os.listdir(self.tmpDir):
#                os.remove(os.path.join(self.tmpDir, f))
#
#            os.rmdir(self.tmpDir)
#
#    def test_convert_traces(self):
#        traces = [
#            ProcessedEvent("oh"),
#            ProcessedEvent("oh"),
#            ProcessedEvent("oh"),
#            OrderedProcessedTraces.endOfTrace(),
#            OrderedProcessedTraces.endOfTrace(),
#            ProcessedEvent("ah"),
#            OrderedProcessedTraces.endOfTrace()]
#
#        class OrderedTraces(OrderedProcessedTraces):
#            def __iter__(self):
#                return traces.__iter__()
#        ordered = OrderedTraces()
#        translated = ProcessedTracesToOrdered(
#            OrderedToProcessedTraces(OrderedTraces()))
#        self.assertEqual(list(ordered), list(translated))
#
#    def test_writing_cache__syntax(self):
#        for traces in [[], [[]], [["\naa", "aa\n"], ["\naa", "aa\n"], []], [[], ["eh"], []], [["""
#
#        oh oh
#
#        ah ah
#        """, "é"]]]:
#            with self.subTest(traces=traces):
#                self.removeCache()
#
#                tl = TracesLoader(TracesProcessor())
#                tl.hardCodedTraces(rawTraces=traces)
#                tl.makeCache(imediate=True)
#
#                tl = TracesLoader(TracesProcessor())
#                tl.hardCodedTraces(rawTraces=traces)
#                self.assertEqual(tl.getTraces().asSimpleList(), traces)
#
#    def test_writing_cache__with_exception(self):
#        traces = [["some", "events"],
#                  ["second", "trace"], []]
#        self.removeCache()
#
#        queue = multiprocessing.SimpleQueue()
#
#        class HCT(TracesProcessor.HardCodedTraces):
#            class Trace(ProcessedTrace):
#                def __init__(self, trace):
#                    self.trace = trace
#
#                def __iter__(self):
#                    for event in self.trace:
#                        yield event
#                        queue.put("event")
#
#            class Traces(ProcessedTraces):
#                def __init__(self, prevTraces):
#                    self.prevTraces = prevTraces
#
#                def __iter__(self):
#                    for trace in self.prevTraces:
#                        yield HCT.Trace(trace)
#                        queue.put("trace")
#
#            def apply(self, traces, rawTraces):
#                return __class__.Traces(ProcessedTraces.fromBase(
#                    TracesProcessor.HardCodedTraces().apply(
#                        traces, rawTraces)))
#
#        def makeTraces(makeCache):
#            processor = TracesProcessor()
#            processor.register(HCT())
#            tl = TracesLoader(processor)
#            tl.hCT(rawTraces=traces)
#            if makeCache:
#                tl.makeCache()
#            return tl
#
#        def makeCache():
#            tl = makeTraces(True)
#            for trace in tl.getTraces():
#                for event in trace:
#                    pass
#                time.sleep(10)
#        process = multiprocessing.Process(target=makeCache)
#
#        process.start()
#        queue.get()
#        os.kill(process.pid, signal.SIGINT)
#        process.join()
#
#        tl = makeTraces(False)
#        self.assertEqual(tl.getTraces().asSimpleList(), traces)
#
#    def test_writing_cache__partial_readers(self):
#
#        def partialReader1(traces):
#            pass
#
#        def partialReader2(traces):
#            for trace in traces:
#                pass
#
#        def partialReader3(traces):
#            """read only first trace"""
#            for trace in traces:
#                for events in trace:
#                    pass
#                break
#
#        def partialReader4(traces):
#            """ read only first event of all traces"""
#            for trace in traces:
#                for event in trace:
#                    break
#
#        def partialReader5(traces):
#            """read only first event of first trace"""
#            for trace in traces:
#                for event in trace:
#                    break
#                break
#
#        def partialReader6(traces):
#            """read events from last trace first"""
#            trace_generators = []
#            for trace in traces:
#                trace_generators.append(trace)
#            for i in range(len(trace_generators) - 1, -1, -1):
#                trace = trace_generators[i]
#                for event in trace:
#                    pass
#
#        def normalReader(traces):
#            for trace in traces:
#                for event in trace:
#                    pass
#        for reader in [
#                partialReader1,
#                partialReader2,
#                partialReader3,
#                partialReader4,
#                partialReader5,
#                partialReader6]:
#            traces = [["some", "simple", "events"],
#                      ["and", "two", "traces"], []]
#            with self.subTest(reader=reader):
#                self.removeCache()
#
#                tl = TracesLoader(TracesProcessor())
#                tl.hardCodedTraces(rawTraces=traces)
#                tl.makeCache()
#                processedTraces = tl.getTraces()
#                reader(processedTraces)
#                processedTraces.load()  # force to read all events and close
#                # the cache file. Otherwise, the cache
#                # file is not written and the traces
#                # are reloaded from HardCodedTraces
#
#                tl = TracesLoader(TracesProcessor())
#                tl.hardCodedTraces(rawTraces=traces)
#                self.assertEqual(tl.getTraces().asSimpleList(), traces)
#
#                tl = TracesLoader(TracesProcessor())
#                tl.hardCodedTraces(rawTraces=traces)
#                reader(tl.getTraces())
#                self.assertEqual(tl.getTraces().asSimpleList(), traces)
#
#    def test_CutByTime(self):
#        def toEvents(seqs):
#            rs = []
#            for seq in seqs:
#                r = []
#                for t in seq:
#                    r.append({'timestamp': t})
#                rs.append(r)
#            return rs
#
#        def testOne(seqs, expected):
#            traces = toEvents(seqs)
#            tl = TracesLoader(TracesProcessor())
#            tl.hardCodedTraces(rawTraces=traces)
#            tl.cutByTime(deltaT=10)
#            self.assertEqual(tl.getTraces().asSimpleList(), toEvents(expected))
#
#        data = [([[1, 2, 3]], [[1, 2, 3]]),
#                ([[1, 2, 15, 16]], [[1, 2], [15, 16]]),
#                ([[1, None, 2, 15, 16]], [[1, None, 2], [15, 16]]),
#                ([[1, 2, None, 15, 16]], [[1, 2, None], [15, 16]]),
#                ([[1, 15], [16]], [[1], [15], [16]]),
#                ([[1, 15], [None]], [[1], [15], [None]]),
#                ([[1, 15], [], [16]], [[1], [15], [], [16]]),
#                ([[1, 15, 123456]], [[1], [15], [123456]]),
#                ([[1], [15]], [[1], [15]]),
#                ]
#
#        for traces, expected in data:
#            with self.subTest(traces=traces):
#                testOne(traces, expected)
#
#    def test_RemoveDouble(self):
#        def testOne(seqs, expected):
#            tl = TracesLoader(TracesProcessor())
#            tl.hardCodedTraces(rawTraces=traces)
#            tl.removeDoubles()
#            self.assertEqual(tl.getTraces().asSimpleList(), expected)
#
#        data = [
#            ([['a', 'b', 'a', 'b'], ['a', 'b', 'a', 'b']],
#             [['a', 'b', 'a', 'b'], ['a', 'b', 'a', 'b']]),
#            ([['a', 'b', 'b']], [['a', 'b']]),
#            ([['a', 'b'], ['b', 'c', 'a']], [['a', 'b'], ['b', 'c', 'a']]),
#            ([['a', 'a', 'a', 'a', 'a'], []], [['a'], []]),
#        ]
#
#        for traces, expected in data:
#            with self.subTest(traces=traces):
#                testOne(traces, expected)
#
#    def test_RemoveDouble(self):
#        def testOne(seqs, expected):
#            tl = TracesLoader(TracesProcessor())
#            tl.hardCodedTraces(rawTraces=traces)
#            tl.removeIncomplete(neededKeys=['a', 'b'])
#            self.assertEqual(tl.getTraces().asSimpleList(), expected)
#
#        data = [
#            ([[{'a': 1, 'b': 1}, {'b': 2, 'a': 3}]], [[{'a': 1, 'b': 1}, {'b': 2, 'a': 3}]]
#             ),
#            ([[{'c': 4, 'b': 5}, {'a': 6, 'b': 6}]], [[{'a': 6, 'b': 6}]]
#             ),
#            ([[{'a': None, 'b': 7}]], [[{'a': None, 'b': 7}]]
#             ),
#            ([[{'a': None}]], [[]]
#             ),
#            ([[{'a': 8, 'b': 9, 'c': 10}], []], [[{'a': 8, 'b': 9, 'c': 10}], []])
#        ]
#
#        for traces, expected in data:
#            with self.subTest(traces=traces):
#                testOne(traces, expected)
#
#    def test_Troncate(self):
#        def testOne(seqs, expected):
#            tl = TracesLoader(TracesProcessor())
#            tl.hardCodedTraces(rawTraces=traces)
#            tl.troncate(length=2)
#            self.assertEqual(tl.getTraces().asSimpleList(), expected)
#
#        data = [
#            ([['a', 'b', 'a', 'b'], ['a', 'b', 'a', 'b']], [['a', 'b'], ['a', 'b']]),
#            ([['a', 'b', 'b']], [['a', 'b']]),
#            ([['a', 'b'], ['b', 'c', 'a']], [['a', 'b'], ['b', 'c']]),
#            ([[1], [], [2, 3, 4]], [[1], [], [2, 3]]),
#        ]
#
#        for traces, expected in data:
#            with self.subTest(traces=traces):
#                testOne(traces, expected)
if __name__ == '__main__':
    unittest.main()
