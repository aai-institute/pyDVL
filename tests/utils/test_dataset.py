""" FIXME: more to do... """

def test_dataset_len(boston_dataset):
    assert len(boston_dataset) == len(boston_dataset.x_train) == 404
    assert len(boston_dataset.x_train) + len(boston_dataset.x_test) == 506

