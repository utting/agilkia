# -*- coding: utf-8 -*-
"""
Data structures for Traces and Sets of Traces.

This defines the 'Event', 'Trace' and 'TraceSet' classes, plus helper functions.

NOTES
=====
1.  *private data fields* (starting with '_') will not be stored in the JSON files.
    For example, each Trace object has a '_parent' point to its TraceSet, but this
    is not stored in the JSON file, since the hierarchical structure of the JSON
    already captures the parent-child relationship between TraceSet and Trace.
2.  *JSON-traces file version numbers* follow the usual Semantic Versioning scheme:
    *(Major.Minor.Patch)*.
    ``TraceSet.upgrade_json_data`` currently just prints a warning message when a
    program running older code reads a JSON file with a newer MINOR version number.
    This allows graceful updating of one program at a time, but does
    have the danger that a older program may read newer data (with a warning),
    then SAVE that data in the slightly older format, thus losing some data.
    But a strict version-equality means that all programs have to be updated
    simultaneously, which is a pain.

Ideas / Tasks to do
===================
    * DONE: add save_as_arff() method like to_pandas.
    * DONE: store event_chars into meta_data.
    * DONE: store signatures into meta_data.
    * DONE: create Event class and make it dict-like.
    * DONE: add support for splitting traces into 'sessions' via splitting or grouping.
    * DONE: add support for clustering traces
    * DONE: add support for visualising the clusters (TSNE).
    * DONE: add 'meta_data' to Trace and Event objects too (replace properties)
    * DONE: add unit tests for clustering...  (Note: not saved in JSON!)
    * DONE: split RandomTester into SmartSequenceGenerator subclass (better meta-data).
    * provide an easy way of copying meta-data across, or cloning a Trace[Set] with new traces.
    * extend to_pandas() to allow user-defined columns to be added.
    * split the test execution methods out of RandomTester into a delegate class,
        so that we can generate tests without executing them, and execute without generating.

@author: utting@usc.edu.au
"""

import os
import sys
from pathlib import Path  # object-oriented filenames!
from collections import defaultdict
import json
import decimal
import datetime
import re
import xml.etree.ElementTree as ET
import pandas as pd            # type: ignore
import numpy as np             # type: ignore
import sklearn.cluster         # type: ignore
import sklearn.preprocessing   # type: ignore
import matplotlib.pyplot as plt
import matplotlib.cm as pltcm
from sklearn.manifold import TSNE
# liac-arff from https://pypi.org/project/liac-arff (via pip)
# import arff                    # type: ignore
from typing import List, Set, Mapping, Dict, Union, Any, Optional, cast


TRACE_SET_VERSION = "0.1.4"

MetaData = Dict[str, Any]


def safe_name(string: str) -> str:
    """Returns 'string' with all non-alpha-numeric characters replaced by '_'."""
    return re.sub("[^A-Za-z0-9]", "_", string)


class Event:
    """An Event is a dictionary-like object that records all the details of an event.

    Public data fields include:
        * self.action (str): the full action name.
        * self.inputs (Dict[str,Any]): the named inputs and their values.
        * self.outputs (Dict[str,Any]): the named outputs and their values.
        * self.meta_data (Dict[str,Any]): any extra properties such as "timestamp".
          Note: if self.meta_data["timestamp"] is present, it should be in ISO 8601 format.
          Or use get_meta(key) to get an individual meta-data value.
    """

    def __init__(self, action: str, inputs: Dict[str, Any], outputs: Dict[str, Any],
                 meta_data: Optional[MetaData] = None):
        self.action = action
        self.inputs = inputs
        self.outputs = outputs
        self.meta_data: MetaData = {} if meta_data is None else meta_data.copy()

    @property
    def status(self) -> int:
        """Read-only status of the operation, where 0 means success.
        If no output 'Status' is available, this method always returns 0.
        """
        return int(self.outputs.get("Status", "0"))

    @property
    def error_message(self) -> str:
        """Read-only error message output by this operation.
        If no output['Error'] field is available, this method always returns "".
        """
        return self.outputs.get("Error", "")

    def __str__(self):
        """Shows action, inputs and outputs, but elides any meta-data."""
        return f"Event({self.action}, {self.inputs}, {self.outputs})"


class Trace:
    """Represents a single trace, which contains a sequence of events.

    Public data fields include:
        * self.events: List[Event].  However, the iteration, indexing, and len(_) methods
          have been lifted from the events list up to this Trace object, so you
          may not need to access self.events at all.
        * self.meta_data: MetaData.  Or use get_meta(key) to get an individual meta-data value.
    """

    def __init__(self, events: List[Event], parent: 'TraceSet' = None,
                 meta_data: Optional[MetaData] = None,
                 random_state=None):
        """Create a Trace object from a list of events.

        Args:
            events: the sequence of Events that make up this trace.
            parent: the TraceSet that this trace is part of.
            meta_data: any meta-data associated with this individual trace.
            random_state: If this trace was generated using some randomness, you should supply
                this optional parameter, to record the state of the random generator at the
                start of the sequence.  For example, rand_state=rand.getstate().
                If this value is supplied, it is added into the meta-data.
        """
        if events and not isinstance(events[0], Event):
            raise Exception("Events required, not: " + str(events[0]) + " ...")
        self.events = events
        self._parent = parent
        self.meta_data: MetaData = {} if meta_data is None else meta_data
        if random_state is not None:
            self.meta_data["random_state"] = random_state

    def trace_set(self):
        """Returns the TraceSet that this trace is part of, or None if not known."""
        return self._parent

    def __iter__(self):
        return self.events.__iter__()

    def __len__(self):
        return len(self.events)

    def __getitem__(self, key):
        return self.events[key]

    def get_meta(self, key: str) -> Optional[Any]:
        """Returns requested meta data, or None if that key does not exist."""
        if key in self.meta_data:
            return self.meta_data[key]
        else:
            return None

    def append(self, event: Event):
        if not isinstance(event, Event):
            raise Exception("Event required, not: " + str(event))
        self.events.append(event)

    def action_counts(self) -> Dict[str, int]:
        """Returns a dictionary of how many times each action occurs in this trace.

        Returns:
            A dictionary of counts that can be used for clustering traces.
        """
        result = defaultdict(int)
        for ev in self.events:
            result[ev.action] += 1
        return result

    def action_status_counts(self) -> Dict[str, int]:
        """Counts how many times each action-status pair occurs in this trace.

        Returns:
            A dictionary of counts that can be used for clustering traces.
        """
        result = defaultdict(int)
        for ev in self.events:
            key = ev.action + "_" + str(ev.status)
            result[key] += 1
        return result

    def to_string(self,
                  to_char: Dict[str, str] = None,
                  compress: List[str] = None,
                  color_status: bool = False):
        """Return a one-line summary of this trace, one character per event.
        See 'trace_to_string' for details.
        NOTE: throws an exception if no to_char map is given and this trace has no parent.
        """
        if to_char is None:
            if self._parent is None:
                raise Exception("Cannot view trace with no parent and no to_char map.")
            to_char = self._parent.get_event_chars()
        return trace_to_string(self.events, to_char, compress=compress, color_status=color_status)

    def __str__(self):
        try:
            return self.to_string()
        except Exception:
            return "???"


class TraceSet:
    """Represents a set of traces, either generated or recorded.

    Invariants:
        * forall tr:self.traces (tr._parent is self)
          (TODO: set _parent to None when a trace is removed?)
        * self.meta_data is a dict with keys: date, source at least.

    Public data fields include:
        * self.traces: List[Trace].  However, the iteration, indexing, and len(_) methods
          have been lifted from the trace list up to the top-level TraceSet object, so you
          may not need to access self.traces at all.
        * self.meta_data: MetaData.  Or use get_meta(key) to get an individual meta-data value.
        * self.version: str.  Version number of this TraceSet object.
    """

    meta_data: MetaData
    _event_chars: Optional[Dict[str, str]]

    def __init__(self, traces: List[Trace], meta_data: Dict[str, Any] = None):
        """Create a TraceSet object from a list of Traces.

        Args:
            traces: the list of Traces.
            meta_data: a dictionary that captures the important meta-data for this set of traces.
                If this TraceSet is to be saved into a file, the meta-data should include at
                least the GDF (General Data Format) compulsory fields,
                which are based on the Dublin Core:
                    "date" in ISO 8601 format;
                    "dataset" for the official name of this TraceSet;
                    "source" for the origin of the dataset;
                    and any other meta-data that is available.
        """
        self.version = TRACE_SET_VERSION
        self.traces = traces
        self._clusters: List[int] = None
        self._cluster_data: pd.DataFrame = None
        trace_parents = set()
        # add all the trace to this set.
        for tr in self.traces:
            if isinstance(tr, Trace):
                if tr._parent and tr._parent != self:
                    trace_parents.add(tr._parent)
                tr._parent = self
            else:
                raise Exception("TraceSet expects List[Trace], not: " + str(type(tr)))
        if meta_data is None:
            if len(trace_parents) == 1:
                # copy across meta-data, since all traces come from same place.
                self.meta_data = next(iter(trace_parents)).meta_data.copy()
                # print("DEBUG: copyied meta-data from traces:", self.meta_data)
                now = datetime.datetime.now().isoformat()
                self.meta_data["date"] = now
            else:
                self.meta_data = self.get_default_meta_data()
        else:
            self.meta_data = meta_data.copy()
        self._event_chars = None  # recalculated if set of traces grows.

    def __iter__(self):
        return self.traces.__iter__()

    def __len__(self):
        return len(self.traces)

    def __getitem__(self, key):
        return self.traces[key]

    def message(self, msg: str):
        """Print a progress message."""
        print("   ", msg)

    @classmethod
    def get_default_meta_data(cls) -> Dict[str, Any]:
        """Generates some basic meta-data such as date, user and command line."""
        now = datetime.datetime.now().isoformat()
        user = os.path.expanduser('~').split('/')[-1]  # usually correct, but can be tricked.
        meta_data: Dict[str, Any] = {
                "date": now,
                "author": user,
                "dataset": "unknown",
                "action_chars": None
                }
        if len(sys.argv) > 0:
            meta_data["source"] = sys.argv[0]  # the path to the running script/tool.
            meta_data["cmdline"] = sys.argv
        return meta_data

    def get_meta(self, key: str) -> Optional[Any]:
        """Returns requested meta data, or None if that key does not exist."""
        if key in self.meta_data:
            return self.meta_data[key]
        else:
            return None

    def set_meta(self, key: str, value:Any) -> Optional[Any]:
        """Sets the requested meta data, and returns the old value if any."""
        old = None
        if key in self.meta_data:
            old = self.meta_data[key]
        self.meta_data[key] = value
        return old

    def append(self, trace: Trace):
        """Appends the given trace into this set of traces.
        This also sets its parent to be this trace set.
        """
        if not isinstance(trace, Trace):
            raise Exception("Trace required, not: " + str(trace))
        trace._parent = self
        self.traces.append(trace)
        self._event_chars = None  # we will recalculate this later

    def set_event_chars(self, given: Mapping[str, str] = None):
        """Sets up the event-to-char map that is used to visualise traces.

        This will calculate a default mapping for any actions that are not in given.
        (See 'default_map_to_chars').

        Args:
            given: optional pre-allocation of a few action names to chars.
            For good readability of the printed traces, it is recommended that extremely
            common actions should be mapped to 'small' characters like '.' or ','.
        """
        if given is None:
            new_given = self.get_meta("action_chars")
        else:
            self.meta_data["action_chars"] = given  # override any previous given map.
            new_given = cast(Dict[str, str], given).copy()  # copy so we don't change orginal.
        actions = all_action_names(self.traces)
        self._event_chars = default_map_to_chars(actions, given=new_given)

    def get_event_chars(self):
        """Gets the event-to-char map that is used to visualise traces.
        This maps each action name to a single character.
        If set_event_chars has not been called, this getter will calculate and cache
        a default mapping from action names to characters.
        """
        if self._event_chars is None:
            self.set_event_chars()
        return self._event_chars

    def __str__(self):
        name = self.meta_data["dataset"]  # required meta data
        return f"TraceSet '{name}' with {len(self)} traces."

    def save_to_json(self, file: Path) -> None:
        if isinstance(file, str):
            print(f"WARNING: converting {file} to Path.  Please learn to speak pathlib.")
            file = Path(file)
        with file.open("w") as output:
            json.dump(self, output, indent=2, cls=TraceEncoder)

    @classmethod
    def load_from_json(cls, file: Path) -> 'TraceSet':
        if isinstance(file, str):
            print(f"WARNING: converting {file} to Path.  Please learn to speak pathlib.")
            file = Path(file)
        if not isinstance(file, Path):
            raise Exception(f"load_from_json requires Path, not {file} (type={type(file)})")
        # with open(filename, "r") as input:
        data = json.loads(file.read_text())
        # Now check version and upgrade if necessary.
        if isinstance(data, list):
            # this file was pre-TraceSet, so just a list of lists of events.
            mtime = datetime.datetime.fromtimestamp(file.stat().st_mtime).isoformat()
            meta = {"date": mtime, "dataset": file.name, "source": "Upgraded from version 0.1"}
            traces = cls([], meta)
            for ev_list in data:
                events = [cls._create_event_object("0.1", ev) for ev in ev_list]
                traces.append(Trace(events))
            return traces
        elif isinstance(data, dict) and data.get("__class__", None) == "TraceSet":
            return cls.upgrade_json_data(data)
        else:
            raise Exception("unknown JSON file format: " + str(data)[0:60])

    @classmethod
    def upgrade_json_data(cls, json_data: Dict) -> 'TraceSet':
        version = json_data["version"]
        if version.startswith("0.1."):
            # This JSON file is compatible with our code.
            # First, convert json_data dicts to Trace and TraceSet objects.
            traceset = TraceSet([], json_data["meta_data"])
            for tr_data in json_data["traces"]:
                traceset.append(cls._create_trace_object(version, tr_data))
            # Next, see if any more little updates are needed.
            if version in ["0.1.2", "0.1.3", TRACE_SET_VERSION]:
                pass  # nothing more to do.
            elif version == "0.1.1":
                # Move given_event_chars into meta_data["action_chars"]
                # Note: traceset["version"] has already been updated to the latest.
                traceset.meta_data["actions_chars"] = json_data["given_event_chars"]
            else:
                # The JSON must be from a newer 0.1.x version, so give a warning.
                print(f"WARNING: reading {version} TraceSet using {TRACE_SET_VERSION} code.")
                print(f"         Some data may be lost.  Please upgrade this program.")
            return traceset
        raise Exception(f"upgrade of TraceSet v{version} to v{TRACE_SET_VERSION} not supported.")

    @classmethod
    def _create_trace_object(cls, version: str, tr_data: Dict[str, Any]) -> Trace:
        assert tr_data["__class__"] == "Trace"
        meta = tr_data.get("meta_data", {})
        rand = tr_data.get("random_state", None)
        if rand is not None:
            meta["random_state"] = rand
        events = [cls._create_event_object(version, ev) for ev in tr_data["events"]]
        return Trace(events, meta_data=meta)

    @classmethod
    def _create_event_object(cls, version: str, ev: Dict[str, Any]) -> Event:
        special = ["action", "inputs", "outputs"]
        action = ev["action"]
        inputs = ev["inputs"]
        outputs = ev["outputs"]
        if version <= "0.1.2":
            props = {key: ev[key] for key in ev if key not in special}
        elif version <= "0.1.3":
            props = ev["properties"]
        else:
            props = ev["meta_data"]
        return Event(action, inputs, outputs, props)

    def to_pandas(self) -> pd.DataFrame:
        """Converts all the traces into a single Pandas DataFrame (one event/row).

        The first three columns are 'Trace' and 'Event' which give the number of the
        trace and the position of the event within that trace, and 'Action' which is
        the name of the action of the event.
        Each named input value is recorded in a separate column.
        For outputs, by default there are just 'Status' (int) and 'Error' (str) columns.
        """
        return traces_to_pandas(self.traces)

    def arff_type(self, pandas_type: str) -> Union[str, List[str]]:
        """Maps each Pandas data type to the closest ARFF type."""
        if pd.api.types.is_integer_dtype(pandas_type):
            return "INTEGER"
        if pd.api.types.is_float_dtype(pandas_type):
            return "REAL"
        if pd.api.types.is_bool_dtype(pandas_type):
            return ["False", "True"]
        return "STRING"
        # TODO: check column to see if NOMINAL is better?
        # raise Exception(f"do not know how to translate Pandas type {pandas_type} to ARFF.")

    def save_to_arff(self, file: Path, name=None) -> None:
        """Save all the events in all traces into an ARFF file for machine learning.

        Args:
            filename: the name of the file to save into.  Should end with '.arff'.
            name: optional relation name to identify this data inside the ARFF file.
                The default is the base name of 'file'.
        """
        if isinstance(file, str):
            print(f"WARNING: converting {file} to Path.  Please learn to speak pathlib.")
            file = Path(file)
        if name is None:
            name = file.stem
        data = self.to_pandas()
        attributes = [(n, self.arff_type(t)) for (n, t) in zip(data.columns, data.dtypes)]
        try:
            import arff
        except ImportError:
            print("Please install ARFF support before using save_to_arff.")
            print("It is a pip only package:  pip install liac-arff")
            return
        with file.open("w") as output:
            contents = {
                "relation": safe_name(name),
                "attributes": attributes,
                "data": data.values,  # [[tr] for tr in trace_summaries],
                "description": "Events from " + name
                }
            arff.dump(contents, output)

    def with_traces_split(self, start_action: str = None, input_name: str = None) -> 'TraceSet':
        """Returns a new TraceSet with each trace in this set split into shorter traces.

        It accepts several split criteria, and will start a new trace whenever any
        of those criteria are true.  At least one criteria must be supplied.

        Args:
            start_action: the name of an action that starts a new trace.
            input_name: the name of an input.  Whenever the value of this input
                changes, then a new trace should be started.  Note that events
                with this input missing are ignored for this splitting criteria.

        Returns:
            a new TraceSet, usually with more traces and shorter traces.
        """
        if start_action is None and input_name is None:
            raise Exception("with_traces_split requires at least one split criteria.")
        traces2 = TraceSet([], self.meta_data)
        # TODO: update meta data with split info?
        for old in self.traces:
            curr_trace = Trace([])
            traces2.append(curr_trace)
            prev_input = None
            for event in old:
                input_value = event.inputs.get(input_name, None)
                input_changed = input_value != prev_input and input_value is not None
                if (event.action == start_action or input_changed) and len(curr_trace) > 0:
                    curr_trace = Trace([])
                    traces2.append(curr_trace)
                curr_trace.append(event)
                if input_value is not None:
                    prev_input = input_value
                # NOTE: we could check end_action here.
        return traces2

    def with_traces_grouped_by(self, name: str, property: bool = False) -> 'TraceSet':
        """Returns a new TraceSet with each trace grouped into shorter traces.

        It generates a new trace for each distinct value of the given input or property name.

        Args:
            name: the name of an input.  A new trace is started for each value of this input
                (or property).  Note that events with this value missing are totally discarded.
            property: True means group by the property called name, rather than an input.

        Returns:
            a new TraceSet, usually with more traces and shorter traces.
        """
        # TODO: update meta data with split info?
        traces2 = TraceSet([], self.meta_data)
        for old in self.traces:
            groups = defaultdict(list)  # for each value this stores a list of Events.
            for event in old:
                if property:
                    value = event.meta_data.get(name, None)
                else:
                    value = event.inputs.get(name, None)
                if value is not None:
                    groups[value].append(event)
            for event_list in groups.values():
                traces2.append(Trace(event_list))
        return traces2

    def get_all_actions(self):
        """Returns a sorted list (with duplicates removed) of all the keys in data."""
        actions = set()
        for tr in self.traces:
            for ev in tr.events:
                actions.add(ev.action)
        return sorted(list(actions))

    def get_trace_data(self, method: str = "action_counts",
                       columns:List[str]=None) -> pd.DataFrame:
        """Returns a Pandas table of statistics/data about each trace.

        This can gather data using any of the zero-parameter data-gathering methods
        of the Trace class.  Any missing data values are replaced by zeroes.

        Note: you can add more data-gathering methods by defining a subclass of Trace
        and using that subclass when you create Trace objects.

        Args:
            method: the name of one of the methods in the Trace objects.
                This method must return a Dict[str, number] for some kind of number.
                The default is the ``action_counts()`` method, which corresponds to the
                *bag-of-words* algorithm.
            columns: optional list of column names.  This can be used to reorder or remove
                or add columns.  (Any added columns will be filled with zeroes).

        Returns:
            A table of data that can be used for clustering or machine learning.
            If columns is not specified, the columns of the table will be in alphabetical order.
            The i'th row of the table is the data for the i'th trace in this set.
        """
        trace_data = [getattr(tr, method)() for tr in self.traces]
        if columns is None:
            # columns is the sorted list of the union of all keys in trace_data.
            keys = set()
            for d in trace_data:
                keys.update(set(d.keys()))
            columns = sorted(list(keys))
        data = pd.DataFrame(trace_data, columns=columns)
        data.fillna(value=0, inplace=True)
        return data

    def create_clusters(self, data: pd.DataFrame, algorithm=None,
                        normalizer=None, fit: bool = True) -> int:
        """Runs a clustering algorithm on the given data and remembers the clusters.

        Note that clustering results are viewed as transient, so are currently not saved
        into JSON files.  Clustering must be repeated after loading a TraceSet.

        Args:
            data: a Pandas DataFrame, typically from get_trace_data(), with the i'th row
                of the DataFrame being for the i'th trace in this set of traces.
            algorithm: a clustering algorithm (default is MeanShift()).
            normalizer: a normalization algorithm (default is MinMaxScaler).
            fit: True means fit the data into clusters, False means just predict clusters
                assuming that the algorithm and normalizer have already been trained.

        Returns:
            The number of clusters generated.
        """
        if algorithm is None:
            if not fit:
                raise Exception("You must supply pre-fitted algorithm when fit=False")
            algorithm = sklearn.cluster.MeanShift()
        if normalizer is None:
            if not fit:
                raise Exception("You must supply pre-fitted normalizer when fit=False")
            normalizer = sklearn.preprocessing.MinMaxScaler()
            # normalizer = sklearn.preprocessing.RobustScaler()

        alg_name = str(algorithm).split("(")[0]
        self.message(f"running {alg_name} on {len(data)} traces.")
        if fit:
            normalizer.fit(data)
        self._cluster_data = pd.DataFrame(normalizer.transform(data), columns=data.columns)
        if fit:
            algorithm.fit(self._cluster_data)
            self._clusters = algorithm.labels_
        else:
            print(" pre predict len=", len(algorithm.labels_))
            self._clusters = algorithm.predict(self._cluster_data)
            print("post predict len=", len(algorithm.labels_), len(self._clusters))
        return max(self._clusters) + 1

    def is_clustered(self) -> bool:
        return self._clusters is not None

    def get_clusters(self) -> List[int]:
        """Get the list of cluster numbers for each trace.

        Precondition: self.is_clustered()
        """
        return self._clusters

    def visualize_clusters(self, algorithm=None, fit: bool = True,
                           xlim=None, ylim=None, cmap=None,
                           markers=None, markersize=None,
                           filename:str = None, block:bool=True):
        """Visualize the clusters from create_clusters().

        Args:
            algorithm: the visualization algorithm to map data into 2D (default TSNE).
            fit: True means fit the data, False means algorithm is pre-trained, so use it
                to just transform the data into 2D without fitting the data first.
                Note that TSNE does not support fit=False yet.
                If you want fit=False, use another dimension-reduction algorithm like PCA(...).
            xlim (Pair[float,float]): optional axis limits for the X axis.
            ylim (Pair[float,float]): optional axis limits for the Y axis.
            cmap (Union[ColorMap,str]): optional color map for the cluster colors,
                or the name of a color map.
                See https://matplotlib.org/3.1.1/tutorials/colors/colormaps.html.
                Default is 'brg', which has a wide range of
                colors going from blue through red to green, and prints in black and white
                okay - though very non-linear - because it does not go all the way to white.
            markers (matplotlib.markers.MarkerStyle): optional marker styles for clusters.
                If this is a string, then the i'th character in the string will be used for
                the i'th marker style.  See https://matplotlib.org/3.1.1/api/markers_api.html
                for the available marker characters.  Note that clusters will be drawn from 0
                up to n-1, so later clusters will be on top.  Also, the earlier clusters tend
                to have more elements.  One approach to improve readability is to use line-based
                shapes (from "1234+x|_") for the first few clusters (which have many points),
                and then filled shapes (from ".o<^>vsphPXd*") for the later clusters
                (which have few points).  Note also that you can use a space for the marker
                character of a cluster if you want to not display that cluster at all.
                However, if your markers string is shorter than the number of clusters,
                all remaining clusters will be displayed using the "o" marker.
            markersize (float): size of the markers in points (only when markers is a str).
                The default seems to be about 6 points.
            filename (str): optional file name to save image into, as well as displaying it.
            block (bool): True (the default) means wait for user to close figure before
                returning.  False means non-blocking.

            Limitations: if you call this multiple times with different numbers of clusters,
                the color map will not be exactly the same.
        """
        data = self._cluster_data
        if data is None or self._clusters is None:
            raise Exception("You must call create_clusters() before visualizing them!")
        num_clusters = max(self._clusters) + 1
        if algorithm is None:
            if not fit:
                raise Exception("You must supply pre-fitted algorithm when fit=False")
            algorithm = TSNE()
        alg_name = str(algorithm).split("(")[0]
        self.message(f"running {alg_name} on {len(data)} traces.")
        if fit:
            tsne_obj = algorithm.fit_transform(data)
        else:
            tsne_obj = algorithm.transform(data)
        # print(tsne_obj[0:5])

        # All the following complex stuff is for adding a 'show label on mouse over' feature
        # to the visualisation scatter graph.
        # It works when run from command line, but not in Jupyter/Spyder!
        # Surely there must be an easier way than doing all this...
        # Code adapted from:
        # https://stackoverflow.com/questions/55891285/how-to-make-labels-appear-
        #     when-hovering-over-a-point-in-multiple-axis/55892690#55892690
        fig, ax = plt.subplots()  # figsize=(8, 6))  # 25% larger, for better printing
        if xlim:
            ax.set_xlim(xlim)
        if ylim:
            ax.set_ylim(ylim)
        if cmap is None:
            # Choose a default colormap.  See bottom of the matplotlib page:
            #   https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
            cmap = pltcm.get_cmap('brg')  # sequential map with nice b&w printing.
        elif isinstance(cmap, str):
            cmap = pltcm.get_cmap(cmap)  # it is the name of a matplotlib color map
        if markers is None:
            markers = "o"
        if isinstance(markers, str) and len(markers) > 1:
            # loop through the marker styles
            clusters = np.ma.array(self._clusters)
            markchars = markers + "o" * num_clusters
            for curr in range(max(num_clusters, len(markers))):
                #prepare for masking arrays - 'conventional' arrays won't do it
                mask = clusters != curr  # False means unmasked
                x_masked = np.ma.array(tsne_obj[:, 0], mask=mask)
                y_masked = np.ma.array(tsne_obj[:, 1], mask=mask)
                color = cmap(curr / num_clusters)
                # c_masked = np.ma.array(clusters, mask=mask)
                # print(f"DEBUG:  mark {curr} is '{markers[curr]}' x={x_masked[0:10]} cl={c_masked[0:10]} color={color}")
                sc = ax.plot(x_masked, y_masked, color=color, linewidth=0,
                             label=f"c{curr}",
                             marker=markchars[curr],
                             markersize=markersize)
            leg = ax.legend(loc='best') #, ncol=2, mode="expand", shadow=True, fancybox=True)
            leg.get_frame().set_alpha(0.5)
        else:
            sc = plt.scatter(tsne_obj[:, 0], tsne_obj[:, 1], c=self._clusters,
                             cmap=cmap, marker=markers)

        if filename:
            plt.savefig(filename)
        names = [str(tr) for tr in self.traces]  # these are in same order as tsne_df rows.

        annot = ax.annotate("",
                            xy=(0, 0),
                            xytext=(0, 20),
                            textcoords="offset points",
                            bbox=dict(boxstyle="round", fc="w"),
                            arrowprops=dict(arrowstyle="->"),
                            )
        annot.set_visible(False)

        def update_annot(ind):
            pos = sc.get_offsets()[ind["ind"][0]]
            annot.xy = pos
            # text = "{}, {}".format(" ".join(list(map(str, ind["ind"]))),
            #                        " ".join([str(names[n]) for n in ind["ind"]]))
            anns = [f"{n} ({self._clusters[n]}): {str(names[n])}" for n in ind["ind"]]
            text = "\n".join(anns)
            annot.set_text(text)
            # annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
            # annot.get_bbox_patch().set_alpha(0.4)

        def hover(event):
            vis = annot.get_visible()
            if event.inaxes == ax:
                cont, ind = sc.contains(event)
                if cont:
                    update_annot(ind)
                    annot.set_visible(True)
                    fig.canvas.draw_idle()
                else:
                    if vis:
                        annot.set_visible(False)
                        fig.canvas.draw_idle()

        fig.canvas.mpl_connect("motion_notify_event", hover)
        plt.show(block=block)

    def get_cluster(self, num: int) -> List[Trace]:
        """Gets a list of all the Trace objects that are in the given cluster."""
        if self._clusters is None:
            raise Exception("You must call create_clusters() before get_cluster(_)")
        if len(self._clusters) != len(self.traces):
            raise Exception("Traces have changed, so you must call create_clusters() again.")
        return [tr for (i, tr) in zip(self._clusters, self.traces) if i == num]


class TraceEncoder(json.JSONEncoder):
    """Custom JSON encoder because objects from zeep could not be serialised.

    Based on ideas from this blog entry by 'The Fellow' (Ouma Rodgers):
    https://medium.com/python-pandemonium/json-the-python-way-91aac95d4041.

    This does not handle XML objects, as they should be decoded via xml_decode first.
    """

    def default(self, obj):
        if isinstance(obj, (dict, list, tuple, str, int, float, bool)):
            return super().default(obj)  # JSON already handles these
        if isinstance(obj, decimal.Decimal):
            return float(round(obj, 6))  # f"{o:.5f}"
        if isinstance(obj, (bytes, bytearray)):
            return "BYTES..."    # TODO: handle these better: repr(o)?
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        if isinstance(obj, (datetime.date, datetime.datetime, datetime.time)):
            return obj.isoformat()  # as a string
        if hasattr(obj, "__dict__"):
            result = {
                "__class__": obj.__class__.__name__,
                "__module__": obj.__module__
                }
            if len(obj.__dict__) == 1 and "__values__" in obj.__dict__:
                # zeep seems to hide the attributes in a __values__ dict.
                # We lift them up to the top level to make the json more readable.
                self._add_public_attributes(result, obj.__dict__["__values__"])
            else:
                self._add_public_attributes(result, obj.__dict__)
            return result
        raise Exception("JSON serialisation not implemented yet for: " +
                        str(obj) + " type " + str(type(obj)) + " dir:" + ",".join(dir(obj)))

    def _add_public_attributes(self, result, attrs) -> None:
        for (name, value) in attrs.items():
            if not name.startswith("_"):
                result[name] = value


def xml_decode(obj: ET.Element) -> Union[str, Dict[str, Any]]:
    """Custom XML encoder to decode XML into a Python dictionary suitable for JSON encoding.

    This roughly follows the ideas from:
    https://www.xml.com/pub/a/2006/05/31/converting-between-xml-and-json.html.

    For simple XML objects with no attributes and no children, this returns just the text string.
    For more complex XML objects, it returns a dictionary.

    Note that the top-level tag of 'obj' is assumed to be handled by the caller.
    That is, the caller will typically do ```d[tag] = xml_decode(obj)``` where xml_decode
    will return either a simple string, or a dictionary.
    """
    if len(obj) == 0 and len(obj.attrib) == 0:
        return cast(str, obj.text)
    else:
        # return obj as a dictionary
        result: Dict[str, Any] = {}
        for (n, v) in obj.attrib.items():
            result[n] = v
        # child objects are more tricky, since some tags may appear multiple times.
        # If a tag appears multiple times, we map it to a list of child objects.
        curr_tag = ""
        curr_list: List[Union[str, Dict[str, Any]]] = []
        for child in obj:
            if child.tag != curr_tag:
                # save the child(ren) we have just finished
                if len(curr_list) > 0:
                    result[curr_tag] = curr_list if len(curr_list) > 1 else curr_list[0]
                curr_list = []
                curr_tag = child.tag
            curr_list.append(xml_decode(child))
        if len(curr_list) > 0:
            result[curr_tag] = curr_list if len(curr_list) > 1 else curr_list[0]
        if obj.text and obj.text.strip():  # ignore text that is just whitespace
            result["text"] = obj.text
        return result


def default_map_to_chars(actions: Set[str], given: Dict[str, str] = None) -> Dict[str, str]:
    """Tries to guess a useful default mapping from action names to single characters.

    Args:
        actions: the names of all the actions.
        given: optional pre-allocation of a few action names to chars.
            You can use this to override the default behaviour.

    Returns:
        A map from every name in actions to a unique single character.
    """
    names: List[str] = sorted(list(actions))
    result: Dict[str, str] = {} if given is None else given.copy()
    # TODO: a better algorithm might be to break up compound words and look for word prefixes?
    curr_prefix = ""
    pass2 = []
    for i in range(len(names)):
        name = names[i]
        if name in result:
            continue  # given
        # skip over any prefix that was in common with previous name.
        if name.startswith(curr_prefix):
            pos = len(curr_prefix)
        else:
            pos = 0
        # check ahead for common prefixes first
        if i + 1 < len(names):
            nxt = names[i + 1]
            if nxt.startswith(name) and name[0] not in result.values():
                result[name] = name[0]
                curr_prefix = name
                continue
            prefix = max([p for p in range(max(len(name), len(nxt))) if name[0:p] == nxt[0:p]])
            # print(f"  found prefix {prefix} of {name} and {nxt}")
            curr_prefix = name[0:prefix]
        else:
            prefix = 0
            curr_prefix = ""
        if prefix > 0 and prefix > pos:
            pos = prefix
        done = False
        for j in range(pos, len(name)):
            if name[pos] not in result.values():
                result[name] = name[pos]
                done = True
                break
        if not done:
            pass2.append(name)
    # Pass 2 (all visible ASCII chars except " and ')
    allchars = "".join([chr(n) for n in range(42, 127)]) + "!#$%&()"
    for name in pass2:
        for ch in name + allchars:
            if ch not in result.values():
                result[name] = ch
                break  # move onto next name in pass2
    return result


def all_action_names(traces: List[Trace]) -> Set[str]:
    """Collects all the action names that appear in the given traces."""
    result = set()
    for tr in traces:
        for ev in tr.events:
            action = ev.action
            result.add(action)
    return result


def trace_to_string(trace: List[Event], to_char: Mapping[str, str], compress: List[str] = None,
                    color_status: bool = False) -> str:
    """Converts a trace to a short summary string, one character per action.

    Args:
        trace: the sequence of JSON-like events, with an "action" field.
        to_char: maps each action name to a single character.  This map must include every
            action name that appears in the traces.  A suitable map can be constructed via
            TraceSet.get_event_chars().
        compress: a list of Action names.  Repeated events will be compressed if in this list.
        color_status: True means color the string red where status is non-zero.
            This uses ANSI escape sequences, so needs to be printed to a terminal.

    Returns:
        a summary string.
    """
    compress_set = set() if compress is None else set(compress)
    chars = []
    prev_action = None
    for ev in trace:
        action = ev.action
        if action == prev_action and action in compress_set:
            # NOTE: we color compressed output just based on the first event.
            pass
        else:
            if color_status and ev.status != 0:
                chars.append("\033[91m")  # start RED
                chars.append(to_char[action])
                chars.append("\033[0m")  # turn off color
            else:
                chars.append(to_char[action])
            prev_action = action
    return "".join(chars)


def traces_to_pandas(traces: List[Trace]) -> pd.DataFrame:
    """Collects all events into a single Pandas DataFrame.

    Columns include the trace number, the event number, the action name, each input parameter,
    the result status and error message.

    TODO: we could convert complex values to strings before sending to Pandas?
    TODO: we could have an option to encode strings into integer properties?
    """
    rows = []
    for tr_num in range(len(traces)):
        events = traces[tr_num].events
        for ev_num in range(len(events)):
            event = events[ev_num]
            row = {"Trace": tr_num, "Event": ev_num, "Action": event.action}
            # we add "Status" and "Error" first, so that those columns come before inputs.
            row["Status"] = event.status
            row["Error"] = event.error_message
            row.update(event.inputs.items())
            rows.append(row)
    return pd.DataFrame(rows)
