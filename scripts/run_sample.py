import sys

sys.path.append(
    "."
)  # needed for importing the config, which is not part of the library
from config import get_config
from valuation.sample_module import SampleClass

if __name__ == "__main__":

    c = get_config()
    print(
        SampleClass().sample_method(
            "Miguel de Benito Delgado <debenito@unternehmertum.de>"
        )
    )
    print("Your new library project valuation is waiting for you!")
    print(f"The related data can be stored in {c.data}")
    print("Try running your first build with tox")
