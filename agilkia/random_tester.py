"""
Simple random test generator for SOAP web services.

Author: Mark Utting, 2019

TODO:
* provide a way of generating related inputs like (lat,long) together.

"""


import requests
import zeep
import zeep.helpers
import getpass
import operator
import random
import numpy
import types
import unittest
from pprint import pprint
from typing import Tuple, List, Mapping


# A signature of a method maps "input"/"output" to the dictionary of input/output names and types.
Signature = Mapping[str, Mapping[str, str]]


# TODO: make these user-configurable
DUMP_WSDL = False         # save each *.wsdl file into current directory.
DUMP_SIGNATURES = True    # save summary of methods into *_signatures.txt
GOOD_PASSWORD = "<GOOD_PASSWORD>"


def summary(value) -> str:
    """Returns a one-line summary of the given value."""
    s = str(value).replace("\n", "").replace(" ", "")
    return s[:60]


def uniq(d):
    """Returns the unique value of a dictionary, else an empty dictionary."""
    result = {}
    for k, v in d.items():
        if result == {} or result == v:
            result = v
            return result  # temp hack - ITM 3 ports have slight differences.
        else:
            print(f"WARNING: uniq sees different values.\n" +
                  " val1={result}\n  val2={v}")
            return {}
    return result


class TestUniq(unittest.TestCase):
    """Some unit tests of the uniq function."""

    def test_normal(self):
        self.assertEqual("def", uniq({"abc": "def"}))

    # TODO: assert uniq({"abc":"one", "xyz":"two"}) == {}

    def test_duplicate_values(self):
        self.assertEquals("one", uniq({"abc": "one", "xyz": "one"}))


def parse_elements(elements):
    """Helper function for build_interface."""
    all_elements = {}
    for name, element in elements:
        all_elements[name] = {}
        all_elements[name]['optional'] = element.is_optional
        if hasattr(element.type, 'elements'):
            all_elements[name]['type'] = parse_elements(element.type.elements)
        else:
            all_elements[name]['type'] = str(element.type)
    return all_elements


def build_interface(client: zeep.Client) -> Mapping[str, Mapping]:
    """Returns a nested dictionary structure for the methods of client.

    Typical usage to get a method called "Login" is:
    ```build_interface(client)[service][port]["operations"]["Login"]```
    """
    interface = {}
    for service in client.wsdl.services.values():
        interface[service.name] = {}
        for port in service.ports.values():
            interface[service.name][port.name] = {}
            operations = {}
            for operation in port.binding._operations.values():
                operations[operation.name] = {}
                operations[operation.name]['input'] = {}
                elements = operation.input.body.type.elements
                operations[operation.name]['input'] = parse_elements(elements)
            interface[service.name][port.name]['operations'] = operations
    return interface


def print_signatures(client: zeep.Client, out):
    """Print a short summary of each operation signature offered by client."""
    # From: https://stackoverflow.com/questions/50089400/introspecting-a-wsdl-with-python-zeep
    for service in client.wsdl.services.values():
        out.write(f"service: {service.name}\n")
        for port in service.ports.values():
            out.write(f"  port: {port.name}\n")
            operations = sorted(
                port.binding._operations.values(),
                key=operator.attrgetter('name'))
            for operation in operations:
                action = operation.name
                inputs = operation.input.signature()
                outputs = operation.output.signature()
                out.write(f"    {action}({inputs})  --> ({outputs})\n")


class RandomTester:
    """Does random testing of a given web service.

    Give it a URL to a server, plus the name(s) of your web service(s),
    and it will read the WSDL specifications from those web services and
    generate any number of random test sequences to test the methods.

    For more sophisticated (user-directed) testing you can also:
    * supply a username and password if login credentials are needed.
    * supply the subset of method names that you want to focus on testing (default is all).
    * supply a set of default input values (or generation functions) for each data type.
    * supply a set of input values (or generation functions) for each named input parameter.
    * TODO: supply a machine learning model for predicting the next best methods to try.
    """
    def __init__(self, base_url, services, methods_to_test=None, input_rules={},
                 rand=random.Random(), verbose=False):
        """Creates a random tester for the server url and set of web services on that server.

        Args:
            base_url (str): the URL to the server that is running the web services.
            services (List[str]): the names of the web services, used to find the WSDL files.
            methods_to_test (List[str]): only these methods will be tested (None means all).
            input_rules (Dict[str,List]): maps each input parameter name to a list of
                possible values, one of which will be chosen randomly.
            rand (random.Random): the random number generator used to generate tests.
            verbose (bool): True means print progress messages during test generation.
        """
        self.base_url = base_url
        self.username = None
        self.password = None
        self.random = rand
        self.verbose = verbose
        self.clients_and_methods = []  # List[(zeep.Service, Dict[str, Signature)]
        self.methods_to_test = methods_to_test
        self.methods_allowed = [] if methods_to_test is None else methods_to_test
        self.named_input_rules = input_rules   # maps each parameter to list of possible 'values'
        self.curr_trace = []
        self.all_traces = [self.curr_trace]
        for w in services:
            self.add_web_service(w)

    def set_username(self, username, password=None):
        """Set the username and (optional) password to be used for the subsequent operations.
        If password is not supplied, this method will immediately interactively prompt for it.
        """
        self.username = username
        self.password = password or getpass.getpass(f"Please enter password for user {username}:")

    def add_web_service(self, name):
        """Add another web service, on the current server."""
        url = self.base_url + "/" + name + ("" if name.upper().endswith("WSDL") else ".asmx?WSDL")
        print("  loading WSDL: ", url)
        if DUMP_WSDL:
            # save the WSDL for reference
            r = requests.get(url, allow_redirects=True)
            open(f"{name}.wsdl", 'wb').write(r.content)
        # now create the client interface for this web service
        client = zeep.Client(wsdl=url)
        interface = build_interface(client)
        pprint([(k, len(v["operations"])) for k, v in uniq(interface).items()])
        if DUMP_SIGNATURES:
            # save summary of this web service into a signatures file
            with open(f"{name}_signatures.txt", "w") as sig:
                print_signatures(client, sig)
        if not uniq(interface):
            print(f"WARNING: web service {name} has empty interface?")
            pprint(interface)
        else:
            ops = uniq(uniq(interface))["operations"]
            self.clients_and_methods.append((client, ops))
            if self.methods_to_test is None:
                self.methods_allowed += list(ops.keys())

    def _find_method(self, name) -> Tuple[zeep.Client, Mapping[str, Signature]]:
        """Find the given method in one of the web services and returns its signature."""
        for (client, interface) in self.clients_and_methods:
            if name in interface:
                return client, interface[name]
        raise Exception(f"could not find {name} in any WSDL specifications.")

    def choose_input_value(self, arg_name: str) -> str:
        """Choose an appropriate value for the input argument called 'arg_name'.
        If no set of input rules is defined for 'arg_name', then 'generate_input_value'
        is called to generate a suitable input value.  Subclasses can override this.

        Args:
            arg_name (str): the name of the input parameter.

        Returns:
            a string if successful, or None if no suitable value was found.
        """
        values = self.named_input_rules.get(arg_name, None)
        if values is None:
            return self.generate_input_value(arg_name)
        val = self.random.choice(values)
        return val

    def generate_input_value(self, arg_name: str) -> any:
        """Can be overridden in subclasses to generate smart values for an input argument."""
        print(f"ERROR: please define possible parameter values for input {arg_name}")
        return None

    def _insert_password(self, arg_value: str) -> str:
        if arg_value == GOOD_PASSWORD:
            return self.password
        else:
            return arg_value

    def get_methods(self) -> Mapping[str, Signature]:
        """Return the set of all method names in all the web services."""
        methods = {}
        for (client, interface) in self.clients_and_methods:
            methods.update(interface)
        return methods

    def call_method(self, name, args=None):
        """Call the web service name(args) and add the result to trace.

        Args:
            name (str): the name of the method to call.
            args (dict): the input values for the method.  If args=None, then this method uses
                :code:choose_input_value to choose appropriate values for each argument
                value of the method.
        Returns:
        Before the call, this method replaces some symbolic inputs by actual concrete values.
        For example the correct password token is replaced by the real password --
        this avoids recording the real password in the inputs of the trace.

        Returns:
            all the data returned by the method.
        """
        (client, signature) = self._find_method(name)
        inputs = signature["input"]
        if args is None:
            args = {n: self.choose_input_value(n) for n in inputs.keys()}
        if self.verbose:
            print(f"    call {name}{args}")
        # insert special secret argument values if requested
        args_list = [self._insert_password(arg) for (n, arg) in args.items()]
        out = getattr(client.service, name)(*args_list)
        # we call it 'action' so it gets printed before 'inputs' (alphabetical order).
        self.curr_trace.append({"action": name, "inputs": args, "outputs": out})
        if self.verbose:
            print(f"    -> {summary(out)}")
        return out

    def generate_trace(self, start=True, length=20, methods=None):
        """Generates the requested length of test steps, choosing methods at random.

        Args:
            start (bool): True means that a new trace is started, beginning with a "Login" call.
            length (int): The number of steps to generate (default=20).
            methods (List[str]): only these methods will be chosen (None means all are allowed)

        Returns:
            the whole of the current trace that has been generated so far.
        """
        if start:
            if self.curr_trace:
                self.curr_trace = []  # start a new trace
                self.all_traces.append(self.curr_trace)
        if methods is None:
            methods = self.methods_allowed
        for i in range(length):  # TODO: continue while Status==0?
            self.call_method(self.random.choice(methods))
        return self.curr_trace

    def setup_feature_data(self):
        """Must be called before the first call to get_trace_features."""
        actions = self.methods_allowed
        nums = len(actions)
        self.action2number = dict(zip(actions, range(nums)))
        self.number2action = dict(zip(range(nums), actions))
        if self.verbose:
            print("Action 2 num:", self.action2number)
            print("Num 2 Action:", self.number2action)

    def get_count_events(self, trace) -> List[int]:
        """Returns an array of counts - how many times each event occurs in trace."""
        result = [0 for k in self.action_number.keys()]
        for ev in trace:
            action_num = self.action2number[ev["action"]]
            result[action_num] += 1
        return result

    def get_trace_features(self) -> List[int]:
        """Returns a vector of numeric features suitable for input to an ML model.
        The results returned by this function must match the training set of the ML model.
        Currently this returns an array of counts - how many times each event occurs
        in the whole current trace, and how many times in the most recent 8 events.
        """
        prefix = self.count_events(self.curr_trace)
        suffix = self.count_events(self.curr_trace[-8:])
        return prefix+suffix

    def generate_trace_ml(self, model, start=True, length=20):
        """Generates the requested length of test steps, choosing methods using the given model.

        Args:
            model (any): the ML model to use to generate the next event.
                This model must support the 'predict_proba' method.
            start (bool): True means that a new trace is started, beginning with a "Login" call.
            length (int): The number of steps to generate (default=20).

        Returns:
            the whole of the current trace that has been generated so far.
        """
        self.setup_feature_data()
        # start a new (empty) trace if requested.
        self.generate_trace(start=start, length=0)
        for i in range(length):
            features = self.get_trace_features(self)
            [proba] = model.predict_proba([features])
            [action_num] = numpy.random.choice(len(proba), p=proba, size=1)
            action = self.number2action[action_num]
            if self.verbose:
                print(i, features, action, ",".join([f"{int(p*100)}" for p in proba]))
            self.call_method(action)
        return self.curr_trace


if __name__ == "__main__":
    unittest.main()
