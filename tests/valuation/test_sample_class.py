from valuation.sample_module import SampleClass


# the suggested naming convention for unit tests is test_method_name_testDescriptionInCamelCase
# this leads to a nicely readable output of pytest
def test_sample_class_attributes_greeterSaysHello():
    greeter = SampleClass()
    assert greeter.hello == "hello "
