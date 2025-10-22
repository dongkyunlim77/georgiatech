import pickle
import unittest

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import torch
from cnn import CNN
from cnn_image_transformations import (
    TransformedDataset,
    create_testing_transformations,
    create_training_transformations,
)
from lstm import LSTM
from NN import NeuralNet as dlnet
from random_forest import RandomForest
from rnn import RNN
from sklearn.metrics import ConfusionMatrixDisplay
from utilities.utils import get_housing_dataset


class TestNN(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.setUp()

        # sample training data
        self.x_train = np.array(
            [
                [
                    0.72176308,
                    0.43961601,
                    0.13553666,
                    0.55713544,
                    0.87702101,
                    0.12019972,
                    0.04842653,
                    0.01573553,
                ],
                [
                    0.53498397,
                    0.81056978,
                    0.17524362,
                    0.00521916,
                    0.2053607,
                    0.90502607,
                    0.99638276,
                    0.45936163,
                ],
                [
                    0.84114195,
                    0.78107371,
                    0.62526833,
                    0.18139081,
                    0.28554493,
                    0.86342263,
                    0.11350829,
                    0.82592072,
                ],
                [
                    0.43286995,
                    0.13815595,
                    0.71456809,
                    0.985452,
                    0.60177364,
                    0.87152055,
                    0.85442663,
                    0.7442592,
                ],
                [
                    0.54714474,
                    0.45039175,
                    0.43588923,
                    0.53943311,
                    0.70734352,
                    0.67388256,
                    0.29136773,
                    0.19560766,
                ],
                [
                    0.5617591,
                    0.86315884,
                    0.34730499,
                    0.13892525,
                    0.53279486,
                    0.79825459,
                    0.37465092,
                    0.23443029,
                ],
                [
                    0.4233198,
                    0.0020612,
                    0.4777035,
                    0.78088463,
                    0.8208675,
                    0.76655747,
                    0.72102559,
                    0.79251294,
                ],
                [
                    0.74503529,
                    0.25137268,
                    0.76440309,
                    0.5790357,
                    0.03791042,
                    0.82510481,
                    0.64463256,
                    0.08997057,
                ],
                [
                    0.81644094,
                    0.51437913,
                    0.75881908,
                    0.96191336,
                    0.56525617,
                    0.70372399,
                    0.75134392,
                    0.56722149,
                ],
            ]
        )
        self.y_train = np.array([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]]).T

    def setUp(self):
        self.nn = dlnet(y=np.random.randn(1, 30), use_dropout=False, use_adam=False)

    def assertAllClose(self, student, truth, msg=None):
        self.assertTrue(np.allclose(student, truth), msg=msg)

    def assertDictAllClose(self, student, truth):
        for key in truth:
            if key not in student:
                self.fail("Key " + key + " missing.")
            self.assertAllClose(student[key], truth[key], msg=(key + " is incorrect."))

        for key in student:
            if key not in truth:
                self.fail("Extra key " + key + ".")

    def test_silu(self):
        u = np.array(
            [
                [
                    1.76405235,
                    0.40015721,
                    0.97873798,
                    2.2408932,
                    1.86755799,
                    -0.97727788,
                ],
                [
                    0.95008842,
                    -0.15135721,
                    -0.10321885,
                    0.4105985,
                    0.14404357,
                    1.45427351,
                ],
                [
                    0.76103773,
                    0.12167502,
                    0.44386323,
                    0.33367433,
                    1.49407907,
                    -0.20515826,
                ],
                [
                    0.3130677,
                    -0.85409574,
                    -2.55298982,
                    0.6536186,
                    0.8644362,
                    -0.74216502,
                ],
                [
                    2.26975462,
                    -1.45436567,
                    0.04575852,
                    -0.18718385,
                    1.53277921,
                    1.46935877,
                ],
            ]
        )
        u_copy = u.copy()
        student = self.nn.silu(u)

        truth = np.array(
            [
                [
                    1.50600053,
                    0.2395843,
                    0.71140327,
                    2.02545844,
                    1.61763291,
                    -0.26721929,
                ],
                [
                    0.68514007,
                    -0.06996226,
                    -0.04894825,
                    0.2468647,
                    0.07719997,
                    1.17891447,
                ],
                [
                    0.51870733,
                    0.06453415,
                    0.27039224,
                    0.19441639,
                    1.22019903,
                    -0.0920934,
                ],
                [
                    0.18083851,
                    -0.25501111,
                    -0.18439194,
                    0.42996694,
                    0.60820579,
                    -0.23937115,
                ],
                [
                    2.05717158,
                    -0.27535592,
                    0.02340263,
                    -0.08485796,
                    1.26057691,
                    1.19452976,
                ],
            ]
        )
        self.assertAllClose(student, truth)
        self.assertAllClose(u, u_copy)
        print_success_message("test_silu")

    def test_d_silu(self):
        u = np.array(
            [
                [
                    1.76405235,
                    0.40015721,
                    0.97873798,
                    2.2408932,
                    1.86755799,
                    -0.97727788,
                ],
                [
                    0.95008842,
                    -0.15135721,
                    -0.10321885,
                    0.4105985,
                    0.14404357,
                    1.45427351,
                ],
                [
                    0.76103773,
                    0.12167502,
                    0.44386323,
                    0.33367433,
                    1.49407907,
                    -0.20515826,
                ],
                [
                    0.3130677,
                    -0.85409574,
                    -2.55298982,
                    0.6536186,
                    0.8644362,
                    -0.74216502,
                ],
                [
                    2.26975462,
                    -1.45436567,
                    0.04575852,
                    -0.18718385,
                    1.53277921,
                    1.46935877,
                ],
            ]
        )
        student = self.nn.derivative_silu(u)
        truth = np.array(
            [
                [
                    1.07401955,
                    0.69486452,
                    0.92117203,
                    1.09858542,
                    1.08265444,
                    0.07927933,
                ],
                [0.91219594, 0.42460936, 0.44848207, 0.69967328, 0.5717735, 1.03387651],
                [0.8467463, 0.56068773, 0.71485409, 0.6637922, 1.04036474, 0.39813594],
                [
                    0.65401393,
                    0.11970306,
                    -0.09884819,
                    0.80494919,
                    0.88386697,
                    0.16036434,
                ],
                [
                    1.09901368,
                    -0.03389202,
                    0.52287128,
                    0.40695176,
                    1.04627518,
                    1.03638487,
                ],
            ]
        )
        self.assertAllClose(student, truth)
        print_success_message("test_d_silu")

    def test_leaky_relu(self):
        alpha = 0.05
        u = np.array(
            [
                [
                    1.76405235,
                    0.40015721,
                    0.97873798,
                    2.2408932,
                    1.86755799,
                    -0.97727788,
                ],
                [
                    0.95008842,
                    -0.15135721,
                    -0.10321885,
                    0.4105985,
                    0.14404357,
                    1.45427351,
                ],
                [
                    0.76103773,
                    0.12167502,
                    0.44386323,
                    0.33367433,
                    1.49407907,
                    -0.20515826,
                ],
                [
                    0.3130677,
                    -0.85409574,
                    -2.55298982,
                    0.6536186,
                    0.8644362,
                    -0.74216502,
                ],
                [
                    2.26975462,
                    -1.45436567,
                    0.04575852,
                    -0.18718385,
                    1.53277921,
                    1.46935877,
                ],
            ]
        )
        u_copy = u.copy()
        student = self.nn.leaky_relu(alpha, u)
        truth = np.array(
            [
                [
                    1.76405235,
                    0.40015721,
                    0.97873798,
                    2.2408932,
                    1.86755799,
                    -0.04886389,
                ],
                [
                    0.95008842,
                    -0.00756786,
                    -0.00516094,
                    0.4105985,
                    0.14404357,
                    1.45427351,
                ],
                [
                    0.76103773,
                    0.12167502,
                    0.44386323,
                    0.33367433,
                    1.49407907,
                    -0.01025791,
                ],
                [
                    0.3130677,
                    -0.04270479,
                    -0.12764949,
                    0.6536186,
                    0.8644362,
                    -0.03710825,
                ],
                [
                    2.26975462,
                    -0.07271828,
                    0.04575852,
                    -0.00935919,
                    1.53277921,
                    1.46935877,
                ],
            ]
        )
        self.assertAllClose(student, truth)
        self.assertAllClose(u, u_copy)
        print_success_message("test_leaky_relu")

    def test_softmax(self):
        input = np.array([[2, 0, 1], [1, 0, 2]])

        actual = self.nn.softmax(input)

        expected = np.array(
            [[0.66524096, 0.09003057, 0.24472847], [0.24472847, 0.09003057, 0.66524096]]
        )

        assert np.allclose(actual, expected, atol=0.1)
        print_success_message("test_softmax")

    def test_d_leaky_relu(self):
        alpha = 0.05
        u = np.array(
            [
                [
                    1.76405235,
                    0.40015721,
                    0.97873798,
                    2.2408932,
                    1.86755799,
                    -0.97727788,
                ],
                [
                    0.95008842,
                    -0.15135721,
                    -0.10321885,
                    0.4105985,
                    0.14404357,
                    1.45427351,
                ],
                [
                    0.76103773,
                    0.12167502,
                    0.44386323,
                    0.33367433,
                    1.49407907,
                    -0.20515826,
                ],
                [
                    0.3130677,
                    -0.85409574,
                    -2.55298982,
                    0.6536186,
                    0.8644362,
                    -0.74216502,
                ],
                [
                    2.26975462,
                    -1.45436567,
                    0.04575852,
                    -0.18718385,
                    1.53277921,
                    1.46935877,
                ],
            ]
        )
        student = self.nn.derivative_leaky_relu(alpha, u)
        truth = np.array(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.05],
                [1.0, 0.05, 0.05, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 0.05],
                [1.0, 0.05, 0.05, 1.0, 1.0, 0.05],
                [1.0, 0.05, 1.0, 0.05, 1.0, 1.0],
            ]
        )
        self.assertAllClose(student, truth)
        print_success_message("test_d_leaky_relu")

    def test_softmax(self):
        input = np.array([[2, 0, 1], [1, 0, 2]])

        actual = self.nn.softmax(input)

        expected = np.array(
            [[0.66524096, 0.09003057, 0.24472847], [0.24472847, 0.09003057, 0.66524096]]
        )

        assert np.allclose(actual, expected, atol=0.1)
        print_success_message("test_softmax")

    def test_d_silu(self):
        u = np.array(
            [
                [
                    1.76405235,
                    0.40015721,
                    0.97873798,
                    2.2408932,
                    1.86755799,
                    -0.97727788,
                ],
                [
                    0.95008842,
                    -0.15135721,
                    -0.10321885,
                    0.4105985,
                    0.14404357,
                    1.45427351,
                ],
                [
                    0.76103773,
                    0.12167502,
                    0.44386323,
                    0.33367433,
                    1.49407907,
                    -0.20515826,
                ],
                [
                    0.3130677,
                    -0.85409574,
                    -2.55298982,
                    0.6536186,
                    0.8644362,
                    -0.74216502,
                ],
                [
                    2.26975462,
                    -1.45436567,
                    0.04575852,
                    -0.18718385,
                    1.53277921,
                    1.46935877,
                ],
            ]
        )
        student = self.nn.derivative_silu(u)
        truth = np.array(
            [
                [
                    1.07401955,
                    0.69486452,
                    0.92117203,
                    1.09858542,
                    1.08265444,
                    0.07927933,
                ],
                [0.91219594, 0.42460936, 0.44848207, 0.69967328, 0.5717735, 1.03387651],
                [0.8467463, 0.56068773, 0.71485409, 0.6637922, 1.04036474, 0.39813594],
                [
                    0.65401393,
                    0.11970306,
                    -0.09884819,
                    0.80494919,
                    0.88386697,
                    0.16036434,
                ],
                [
                    1.09901368,
                    -0.03389202,
                    0.52287128,
                    0.40695176,
                    1.04627518,
                    1.03638487,
                ],
            ]
        )
        self.assertAllClose(student, truth)
        print_success_message("test_d_silu")

    def test_dropout(self):
        np.random.seed(0)
        u = np.array(
            [
                [
                    1.76405235,
                    0.40015721,
                    0.97873798,
                    2.2408932,
                    1.86755799,
                    -0.97727788,
                ],
                [
                    0.95008842,
                    -0.15135721,
                    -0.10321885,
                    0.4105985,
                    0.14404357,
                    1.45427351,
                ],
                [
                    0.76103773,
                    0.12167502,
                    0.44386323,
                    0.33367433,
                    1.49407907,
                    -0.20515826,
                ],
                [
                    0.3130677,
                    -0.85409574,
                    -2.55298982,
                    0.6536186,
                    0.8644362,
                    -0.74216502,
                ],
                [
                    2.26975462,
                    -1.45436567,
                    0.04575852,
                    -0.18718385,
                    1.53277921,
                    1.46935877,
                ],
            ]
        )
        student, _ = self.nn._dropout(u, prob=0.3)

        truth = np.array(
            [
                [2.52007479, 0.57165316, 1.39819711, 3.201276, 2.66793999, -1.39611126],
                [
                    1.35726917,
                    -0.21622459,
                    -0.1474555,
                    0.58656929,
                    0.20577653,
                    2.07753359,
                ],
                [1.08719676, 0.17382146, 0.0, 0.0, 0.0, -0.29308323],
                [
                    0.44723957,
                    -1.22013677,
                    -3.64712831,
                    0.93374086,
                    1.23490886,
                    -1.06023574,
                ],
                [0.0, -2.07766524, 0.0, -0.2674055, 2.18968459, 2.09908396],
            ]
        )

        self.assertAllClose(student, truth)
        print_success_message("test_dropout")

    def test_loss(self):
        y = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 0, 0], [0, 0, 1]])

        # Model's predicted probabilities for each class
        yh = np.array(
            [
                [0.8, 0.15, 0.05],
                [0.1, 0.7, 0.2],
                [0.05, 0.1, 0.85],
                [0.9, 0.05, 0.05],
                [0.1, 0.3, 0.6],
            ]
        )

        # Calculate Cross-Entropy
        student = self.nn.cross_entropy_loss(y, yh)

        truth = 0.2717047128349055

        self.assertAllClose(student, truth)
        print_success_message("test_loss")

    def test_forward_without_dropout(self):
        # load nn parameters
        file = open("data/test_data/nn_param_new.pickle", "rb")
        self.nn.init_parameters(param=pickle.load(file))
        file.close()

        np.random.seed(42)

        x = np.random.rand(10, 3)

        student = self.nn.forward(x, use_dropout=False)

        truth = np.array(
            [
                [0.3958978, 0.30584318, 0.29825903],
                [0.43144733, 0.29665991, 0.27189276],
                [0.33963289, 0.32796592, 0.33240119],
                [0.52733806, 0.23615864, 0.2365033],
                [0.50054525, 0.2646651, 0.23478965],
                [0.35803046, 0.31989139, 0.32207815],
                [0.40936872, 0.29854795, 0.29208333],
                [0.34765758, 0.32675282, 0.32558959],
                [0.38438339, 0.32301924, 0.29259736],
                [0.39337298, 0.31998409, 0.28664292],
            ]
        )

        self.assertAllClose(student, truth)
        print_success_message("test_forward_without_dropout")

    def test_forward_with_dropout(self):
        # control random seed
        np.random.seed(0)

        # load nn parameters
        file = open("data/test_data/nn_param_new.pickle", "rb")
        self.nn.init_parameters(param=pickle.load(file))
        file.close()

        np.random.seed(42)

        x = np.random.rand(10, 3)

        student = self.nn.forward(x, use_dropout=True)

        truth = np.array(
            [
                [0.390886, 0.35127845, 0.25783555],
                [0.49026527, 0.282568, 0.22716673],
                [0.35001604, 0.30759433, 0.34238963],
                [0.51959078, 0.18684784, 0.29356138],
                [0.60251945, 0.21274272, 0.18473783],
                [0.35463254, 0.34448206, 0.3008854],
                [0.37801691, 0.29053599, 0.3314471],
                [0.34559356, 0.33068951, 0.32371693],
                [0.40047828, 0.33824323, 0.2612785],
                [0.37541583, 0.30816614, 0.31641802],
            ]
        )

        self.assertAllClose(student, truth)
        print_success_message("test_forward")

    def test_compute_gradients_without_dropout(self):
        # Load updated parameters
        nn_param = open("data/test_data/nn_param_new.pickle", "rb")
        self.nn.init_parameters(param=pickle.load(nn_param))
        nn_param.close()

        # Load updated cache
        cache = open("data/test_data/test_compute_gradients_cache_new.pickle", "rb")
        self.nn.cache = pickle.load(cache)
        cache.close()

        # Transpose cache data as needed
        for p in self.nn.cache:
            self.nn.cache[p] = self.nn.cache[p]

        # Update y and yh to new dimensions (match the network architecture)
        y = np.array(
            [
                [1, 0, 0],  # Class 0 for Sample 1
                [0, 1, 0],  # Class 1 for Sample 2
                [0, 0, 1],  # Class 2 for Sample 3
                [1, 0, 0],  # Class 0 for Sample 4
                [0, 1, 0],  # Class 1 for Sample 5
                [1, 0, 0],  # Class 0 for Sample 6
                [0, 0, 1],  # Class 2 for Sample 7
                [0, 1, 0],  # Class 1 for Sample 8
                [1, 0, 0],  # Class 0 for Sample 9
                [0, 0, 1],  # Class 2 for Sample 10
            ]
        )

        yh = np.array(
            [
                [0.45114982, 0.27534103, 0.27350915],
                [0.50694454, 0.25828013, 0.23477533],
                [0.376928, 0.31075548, 0.31231653],
                [0.64020962, 0.17592685, 0.18386353],
                [0.60049779, 0.21113381, 0.1883684],
                [0.38496115, 0.30412801, 0.31091084],
                [0.47722575, 0.26188546, 0.26088879],
                [0.36382012, 0.31778386, 0.31839602],
                [0.42644384, 0.30231422, 0.27124194],
                [0.44405581, 0.29595013, 0.25999406],
            ]
        )  # (10,3)

        # Compute the gradients without dropout
        student = self.nn.compute_gradients(y, yh)
        # print(student)
        # print("theta3 gradient shape:", student["theta3"].shape)

        # Update the expected truth values for theta1, theta2, theta3, and biases
        truth = {
            "theta1": np.array(
                [
                    [0.00380808, -0.00636651, -0.01962059, 0.01891073, 0.0125835],
                    [-0.00499717, -0.0022194, 0.00684607, 0.00458131, -0.01015522],
                    [0.01135327, -0.00668598, -0.02584081, 0.02155909, 0.02047891],
                ]
            ),
            "b1": np.array(
                [0.01433871, -0.01029441, -0.02608798, 0.03391497, 0.0107943]
            ),
            "theta2": np.array(
                [
                    [0.0068326, 0.02035328, -0.01197735, -0.00453211, -0.00091109],
                    [0.0059458, 0.00814972, -0.02538777, -0.0040607, 0.00226478],
                    [0.00254007, 0.01351559, -0.02919927, -0.0000304, 0.00623339],
                    [0.0118008, 0.05877338, -0.05721963, 0.01208441, 0.01575452],
                    [0.00616151, 0.01772823, -0.03475608, 0.00041876, 0.00874516],
                ]
            ),
            "b2": np.array(
                [0.01672686, 0.04510498, -0.03001964, -0.00883524, -0.00118399]
            ),
            "theta3": np.array(
                [
                    [0.11512291, -0.0211416, -0.09398131],
                    [-0.00472817, -0.01578938, 0.02051755],
                    [0.03034595, -0.02774979, -0.00259616],
                    [0.0655147, 0.00383751, -0.06935222],
                    [-0.02385389, 0.02962021, -0.00576631],
                ]
            ),
            "b3": np.array([0.06722364, -0.0286501, -0.03857354]),
        }

        # Compare the computed gradients with the truth
        self.assertDictAllClose(student, truth)
        print_success_message("test_compute_gradients_without_dropout")

    def test_compute_gradients_with_dropout(self):
        nn = dlnet(y=np.random.randn(1, 30), use_dropout=True, use_adam=False)
        nn_param = open("data/test_data/nn_param_new.pickle", "rb")
        nn.init_parameters(param=pickle.load(nn_param))
        nn_param.close()

        cache = open("data/test_data/test_compute_gradients_cache_new.pickle", "rb")
        nn.cache = pickle.load(cache)
        cache.close()

        for p in nn.cache:
            nn.cache[p] = nn.cache[p]

        y = np.array(
            [
                [1, 0, 0],  # Class 0 for Sample 1
                [0, 1, 0],  # Class 1 for Sample 2
                [0, 0, 1],  # Class 2 for Sample 3
                [1, 0, 0],  # Class 0 for Sample 4
                [0, 1, 0],  # Class 1 for Sample 5
                [1, 0, 0],  # Class 0 for Sample 6
                [0, 0, 1],  # Class 2 for Sample 7
                [0, 1, 0],  # Class 1 for Sample 8
                [1, 0, 0],  # Class 0 for Sample 9
                [0, 0, 1],  # Class 2 for Sample 10
            ]
        )
        yh = np.array(
            [
                [0.44241676, 0.33622821, 0.22135503],
                [0.60897122, 0.22194954, 0.16907924],
                [0.36570828, 0.2890999, 0.34519182],
                [0.62958633, 0.12340082, 0.24701284],
                [0.74252438, 0.13642363, 0.12105198],
                [0.37250028, 0.35025135, 0.27724837],
                [0.41267473, 0.2645114, 0.32281387],
                [0.35629951, 0.3272992, 0.31640129],
                [0.44909499, 0.32278295, 0.22812206],
                [0.41330537, 0.28034051, 0.30635412],
            ]
        )
        student = nn.compute_gradients(y, yh)
        # print(student)

        truth = {
            "theta1": np.array(
                [
                    [0.02300571, -0.00418651, -0.00911002, 0.02604896, -0.02233272],
                    [0.00104165, -0.00036463, 0.01426645, 0.00899406, -0.01644332],
                    [0.01275034, -0.0053495, -0.01128532, 0.02307632, -0.01482447],
                ]
            ),
            "b1": np.array(
                [0.05580489, -0.00471401, 0.01724874, 0.04352408, -0.04208113]
            ),
            "theta2": np.array(
                [
                    [0.00811666, 0.02846524, -0.01306624, -0.00528399, -0.00063271],
                    [0.00627457, 0.01136866, -0.02554099, -0.00416751, 0.00299897],
                    [0.00396688, 0.02188122, -0.02961279, -0.00119265, 0.00608747],
                    [0.01369191, 0.07259933, -0.06249332, 0.01290772, 0.01768601],
                    [0.0075794, 0.02312872, -0.03147214, -0.00182617, 0.00735683],
                ]
            ),
            "b2": np.array(
                [0.0185258, 0.05827111, -0.03317586, -0.00889289, 0.00054591]
            ),
            "theta3": np.array(
                [
                    [0.11615164, -0.02704535, -0.08910628],
                    [0.00698212, -0.01659459, 0.00961246],
                    [0.03556941, -0.02808471, -0.0074847],
                    [0.0702694, 0.00220021, -0.07246961],
                    [-0.02613038, 0.0342784, -0.00814802],
                ]
            ),
            "b3": np.array([0.07930819, -0.03477125, -0.04453694]),
        }

        self.assertDictAllClose(student, truth)
        print_success_message("test_compute_gradients_with_dropout")

    def test_update_weights(self):
        nn_param = open("data/test_data/nn_param_new.pickle", "rb")
        self.nn.init_parameters(param=pickle.load(nn_param))
        nn_param.close()

        dLoss_file = open("data/test_data/dLoss_new.pickle", "rb")
        dLoss = pickle.load(dLoss_file)
        dLoss_file.close()

        # for p in dLoss:
        #     dLoss[p] = dLoss[p].T

        self.nn.update_weights(dLoss)
        student = self.nn.parameters

        theta1 = np.array(
            [
                [1.00083557, 0.2270293, 0.55528726, 1.27137136, 1.05955953],
                [-0.55445887, 0.53903292, -0.08587255, -0.05856124, 0.23295317],
                [0.08172316, 0.82508247, 0.43177496, 0.06903235, 0.25182592],
            ]
        )

        b1 = np.array([-0.00333674, -0.01494079, 0.00205158, -0.00313068, 0.00854096]).T

        theta2 = np.array(
            [
                [0.17475359, 0.66163629, -0.10039393, 0.14742978, -0.40466077],
                [-1.1271881, 0.29184954, 0.38845946, -0.34723408, 1.00037154],
                [-0.65196158, 0.01668221, -0.07483331, 0.70528767, 0.66059634],
                [0.06773111, 0.15681651, -0.40905365, -0.88196584, -0.15256802],
                [0.08040691, 0.5644029, 0.55478332, -0.19272557, -0.13009738],
            ]
        )

        b2 = np.array([0.00438074, 0.01252795, -0.0077749, 0.01613898, 0.0021274]).T

        theta3 = np.array(
            [
                [-0.45997248, -0.63892035, -0.75795918],
                [0.8842196, -0.22764156, -0.2001961],
                [-0.56093229, 0.34467954, -0.71541384],
                [-0.09151293, -0.39374022, 0.17662359],
                [-0.22030754, -0.51073194, -0.01437774],
            ]
        )
        b3 = np.array([0.00401781, 0.01630198, -0.00462782]).T

        truth = {
            "theta1": theta1,
            "b1": b1,
            "theta2": theta2,
            "b2": b2,
            "theta3": theta3,
            "b3": b3,
        }

        self.assertDictAllClose(student, truth)
        print_success_message("test_update_weights")

    def test_update_weights_with_adam(self):
        nn = dlnet(y=np.random.randn(1, 30), use_dropout=True, use_adam=True)
        nn_param = open("data/test_data/nn_param_new.pickle", "rb")
        nn.init_parameters(param=pickle.load(nn_param))
        nn_param.close()

        dLoss_file = open("data/test_data/dLoss_new.pickle", "rb")
        dLoss = pickle.load(dLoss_file)
        dLoss_file.close()

        nn.update_weights(dLoss)
        student = nn.parameters
        # print(student)

        theta1 = np.array(
            [
                [1.0084761, 0.22103087, 0.55507464, 1.28378029, 1.06823511],
                [-0.55423165, 0.5385338, -0.07738613, -0.04959343, 0.22705916],
                [0.0731636, 0.8296252, 0.42938534, 0.0602491, 0.24626456],
            ]
        )

        b1 = np.array([-0.01, -0.01, 0.01, -0.01, 0.01]).T

        theta2 = np.array(
            [
                [0.1592237, 0.65817247, -0.10174956, 0.15000813, -0.39196323],
                [-1.13173175, 0.28230712, 0.39658762, -0.34190629, 1.00506513],
                [-0.6604121, 0.01046383, -0.07371116, 0.6954797, 0.66711722],
                [0.0592946, 0.15911942, -0.40702986, -0.87583911, -0.14559104],
                [0.07992138, 0.56020272, 0.54772062, -0.18321782, -0.1251939],
            ]
        )

        b2 = np.array([0.01, 0.01, -0.01, 0.01, 0.01]).T

        theta3 = np.array(
            [
                [-0.45892714, -0.64505133, -0.75306723],
                [0.88241328, -0.21792339, -0.20591278],
                [-0.57026712, 0.33770426, -0.71175706],
                [-0.08514035, -0.39046482, 0.18302806],
                [-0.218439, -0.51799476, -0.02260348],
            ]
        )

        b3 = np.array([0.01, 0.01, -0.01]).T

        truth = {
            "theta1": theta1,
            "b1": b1,
            "theta2": theta2,
            "b2": b2,
            "theta3": theta3,
            "b3": b3,
        }

        self.assertDictAllClose(student, truth)
        print_success_message("test_update_weights_with_adam")

    def test_backward(self):
        nn = dlnet(y=np.random.randn(1, 30), use_dropout=True, use_adam=False)
        nn_param = open("data/test_data/nn_param_new.pickle", "rb")
        nn.init_parameters(param=pickle.load(nn_param))
        nn_param.close()

        cache = open("data/test_data/test_compute_gradients_cache_new.pickle", "rb")
        nn.cache = pickle.load(cache)
        cache.close()

        for p in nn.cache:
            nn.cache[p] = nn.cache[p]

        y = np.array(
            [
                [1, 0, 0],  # Class 0 for Sample 1
                [0, 1, 0],  # Class 1 for Sample 2
                [0, 0, 1],  # Class 2 for Sample 3
                [1, 0, 0],  # Class 0 for Sample 4
                [0, 1, 0],  # Class 1 for Sample 5
                [1, 0, 0],  # Class 0 for Sample 6
                [0, 0, 1],  # Class 2 for Sample 7
                [0, 1, 0],  # Class 1 for Sample 8
                [1, 0, 0],  # Class 0 for Sample 9
                [0, 0, 1],  # Class 2 for Sample 10
            ]
        )
        yh = np.array(
            [
                [0.44241676, 0.33622821, 0.22135503],
                [0.60897122, 0.22194954, 0.16907924],
                [0.36570828, 0.2890999, 0.34519182],
                [0.62958633, 0.12340082, 0.24701284],
                [0.74252438, 0.13642363, 0.12105198],
                [0.37250028, 0.35025135, 0.27724837],
                [0.41267473, 0.2645114, 0.32281387],
                [0.35629951, 0.3272992, 0.31640129],
                [0.44909499, 0.32278295, 0.22812206],
                [0.41330537, 0.28034051, 0.30635412],
            ]
        )
        nn.backward(y, yh)
        student = nn.parameters

        expected_theta1 = np.array(
            [
                [1.01824604, 0.23107274, 0.56516574, 1.2935198, 1.07845844],
                [-0.56424206, 0.54853745, -0.08752879, -0.05968337, 0.23722359],
                [0.08303609, 0.8396787, 0.43949819, 0.07001834, 0.2564128],
            ]
        )

        expected_b1 = np.array(
            [
                -5.58048878e-04,
                4.71400839e-05,
                -1.72487378e-04,
                -4.35240822e-04,
                4.20811313e-04,
            ]
        )

        expected_theta2 = np.array(
            [
                [0.14914253, 0.66788782, -0.0916189, 0.14006097, -0.3819569],
                [-1.1417945, 0.29219344, 0.38684303, -0.33186461, 1.01503514],
                [-0.65045177, 0.02024502, -0.08341503, 0.68549163, 0.65705634],
                [0.06915768, 0.16839343, -0.39640492, -0.88596819, -0.1557679],
                [0.06984559, 0.54997143, 0.53803534, -0.17319956, -0.13526747],
            ]
        )

        expected_b2 = np.array(
            [
                -1.85257953e-04,
                -5.82711058e-04,
                3.31758572e-04,
                8.89289448e-05,
                -5.45912211e-06,
            ]
        )

        expected_theta3 = np.array(
            [
                [-0.47008866, -0.63478087, -0.76217616],
                [0.87234346, -0.22775744, -0.19600891],
                [-0.56062281, 0.3479851, -0.72168221],
                [-0.09584304, -0.40048682, 0.17375275],
                [-0.2281777, -0.52833755, -0.012522],
            ]
        )

        expected_b3 = np.array([-0.00079308, 0.00034771, 0.00044537])

        expected = {
            "theta1": expected_theta1,
            "b1": expected_b1,
            "theta2": expected_theta2,
            "b2": expected_b2,
            "theta3": expected_theta3,
            "b3": expected_b3,
        }
        self.assertDictAllClose(student, expected)
        print_success_message("test_backward")

    def test_gradient_descent(self):
        x_train, y_train, x_test, y_test = get_housing_dataset()

        nn = dlnet(y_train, lr=0.01, use_dropout=False, use_adam=False)

        nn.gradient_descent(x_train, y_train, iter=3, local_test=True)

        gd_loss = np.array([1.086892, 1.086749, 1.086607])
        gd_loss_test = np.allclose(np.array(nn.loss), gd_loss, rtol=1e-1)
        print("\nYour GD losses works within the expected range:", gd_loss_test)
        self.assertTrue(gd_loss_test)

    # def test_stochastic_gradient_descent(self):
    #     x_train, y_train, x_test, y_test = get_housing_dataset()

    #     nn = dlnet(y_train, lr=0.01, use_dropout=False, use_adam=False)

    #     nn.stochastic_gradient_descent(x_train, y_train, iter=3, local_test=True)

    #     gd_loss = np.array([1.030836, 1.090258, 1.210505])
    #     gd_loss_test = np.allclose(np.array(nn.loss), gd_loss, rtol=1e-1)
    #     print("\nYour SGD losses works within the expected range:", gd_loss_test)

    def test_minibatch_gradient_descent(self):
        x_train, y_train, x_test, y_test = get_housing_dataset()

        np.random.seed(0)
        nn = dlnet(y_train, lr=0.01, batch_size=6, use_dropout=False, use_adam=False)

        bgd_loss = np.array([1.085044, 1.110789, 1.109005])

        batch_y = np.array(
            [
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                1,
                1,
                0,
                0,
                0,
                0,
                1,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                1,
                0,
                1,
                0,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                1,
                0,
            ]
        )
        batch_y = batch_y.reshape((3, 6, 3))

        nn.minibatch_gradient_descent(x_train, y_train, iter=3, local_test=True)

        # batch_str = "batch_y at iteration %i: "
        # print("\ny_train input:", y_train)
        # [print(batch_str % (i), batch_y) for i, batch_y in enumerate(nn.batch_y)]

        batch_y_test = np.allclose(np.array(nn.batch_y), batch_y, rtol=1e-1)
        print("Your batch_y works within the expected range:", batch_y_test)

        bgd_loss_test = np.allclose(np.array(nn.loss), bgd_loss, rtol=1e-2)
        print(
            "\nYour mini-batch GD losses works within the expected range:",
            bgd_loss_test,
        )
        self.assertTrue(bgd_loss_test)

    def test_gradient_descent_with_adam(self):
        gd_loss_with_adam = [1.086276, 1.080218, 1.074970]
        np.random.seed(0)
        x_train, y_train, x_test, y_test = get_housing_dataset()
        nn = dlnet(y_train, lr=0.01, use_dropout=False, use_adam=True)
        nn.gradient_descent(x_train, y_train, iter=3, local_test=True)
        gd_loss_test_with_adam = np.allclose(
            np.array(nn.loss), gd_loss_with_adam, rtol=1e-2
        )
        print(
            "\nYour GD losses works within the expected range:",
            gd_loss_test_with_adam,
        )
        self.assertTrue(gd_loss_test_with_adam)

    def test_minibatch_gradient_descent_with_adam(self):
        np.random.seed(0)
        x_train, y_train, x_test, y_test = get_housing_dataset()

        nn = dlnet(y_train, lr=0.01, use_dropout=False, use_adam=True)

        nn.minibatch_gradient_descent(x_train, y_train, iter=3, local_test=True)

        gd_loss = np.array([1.080808, 1.095893, 1.071157])
        gd_loss_test = np.allclose(np.array(nn.loss), gd_loss, rtol=1e-2)
        print("\nYour GD losses works within the expected range:", gd_loss_test)
        self.assertTrue(gd_loss_test)


class TestCNN(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.student_cnn = CNN()

    def test_model_architecture(self):
        arbitrary_input = torch.randn((1, 1, 84, 84))
        try:
            output = self.student_cnn.forward(arbitrary_input)
            output = output.squeeze()
            self.assertTrue(
                output.ndim == 1 and output.shape[0] == 4,
                f"Expected output to contain 4 classes. Yours contains {output.shape[0]} classes.",
            )
        except RuntimeError as e:
            if "size mismatch" in str(e) or "shape" in str(e):
                self.assertTrue(False, f"Model hidden layers incompatible: {str(e)}")
            else:
                self.assertTrue(False, f"Runtime Error: {str(e)}")
        print_success_message("test_model_architecture")

    def test_cnn_train_loss_plot(self, trainer):
        (
            train_loss,
            train_acc,
            test_loss,
            test_acc,
        ) = trainer.get_training_history()

        self.assertTrue(
            len(train_loss) == len(train_acc), "len(train_loss) != len(train_acc)"
        )

        THRESHOLD = 3
        increasing_edges = 0
        decreasing_edges = 0

        N = 3
        # Check if train loss is decreasing and train accuracy is increasing
        for i in range(len(train_loss) - N):
            j = i + N

            if train_loss[j] > train_loss[i]:
                increasing_edges += 1

            if train_acc[j] < train_acc[i]:
                decreasing_edges += 1

        self.assertTrue(
            increasing_edges < THRESHOLD,
            f"In train loss plot: {increasing_edges} increasing edges >= {THRESHOLD} threshold\n{train_loss}",
        )
        self.assertTrue(
            decreasing_edges < THRESHOLD,
            f"In train accuracy plot: {decreasing_edges} decreasing edges >= {THRESHOLD} threshold\n{train_acc}",
        )
        print_success_message("test_cnn_train_loss_plot")

    def test_cnn_test_loss_plot(self, trainer):
        (
            train_loss,
            train_acc,
            test_loss,
            test_acc,
        ) = trainer.get_training_history()

        self.assertTrue(
            len(test_loss) == len(test_acc), "len(train_loss) != len(train_acc)"
        )

        THRESHOLD = 3
        increasing_edges = 0
        decreasing_edges = 0
        N = 3

        # Check if train loss is decreasing and train accuracy is increasing
        for i in range(len(test_loss) - N):
            j = i + N

            if test_loss[j] > test_loss[i]:
                increasing_edges += 1

            if test_acc[j] < test_acc[i]:
                decreasing_edges += 1

        self.assertTrue(
            increasing_edges < THRESHOLD,
            f"In test loss plot: {increasing_edges} increasing edges >= {THRESHOLD} threshold\n{test_loss}",
        )
        self.assertTrue(
            decreasing_edges < THRESHOLD,
            f"In test accuracy plot: {decreasing_edges} decreasing edges >= {THRESHOLD} threshold\n{test_acc}",
        )
        print_success_message("test_cnn_test_loss_plot")

    def test_cnn_confusion_matrix(self, trainer, testloader):
        classes = ["glioma", "meningioma", "notumor", "pituitary"]
        y_pred, y_pred_classes, y_gt_classes = trainer.predict(testloader)
        y_pred_prob = torch.max(y_pred, dim=1).values
        # print(f"Test accuracy: {accuracy_score(y_gt_classes, y_pred_classes)}")

        num_classes = len(torch.unique(y_gt_classes))
        correct_per_class = torch.zeros(num_classes)
        total_per_class = torch.zeros(num_classes)

        # Count correct predictions and total samples for each class
        for pred, gt in zip(y_pred_classes, y_gt_classes):
            total_per_class[gt] += 1
            if pred == gt:
                correct_per_class[gt] += 1

        # Calculate accuracy percentage for each class
        accuracy_per_class = torch.where(
            total_per_class > 0,
            (correct_per_class / total_per_class),
            torch.zeros_like(total_per_class),
        )
        ACCURACY_THRESHOLD = 0.5
        # print(accuracy_per_class)
        ConfusionMatrixDisplay.from_predictions(
            y_gt_classes, y_pred_classes, normalize="true", display_labels=classes
        )
        plt.show()
        self.assertTrue(
            (accuracy_per_class >= ACCURACY_THRESHOLD).all(),
            f"In confusion matrix, diagonal entries should be >= {ACCURACY_THRESHOLD}",
        )
        print_success_message("test_cnn_confusion_matrix")


class TestRandomForest(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)

    def test_bootstrapping(self):
        test_seed = 1
        num_feats = 40
        max_feats = 0.65
        rf_test = RandomForest(4, 5, max_feats)

        row_idx, col_idx = rf_test._bootstrapping(15, num_feats, test_seed)
        assert np.array_equal(
            row_idx, np.array([5, 11, 12, 8, 9, 11, 5, 0, 0, 1, 12, 7, 13, 12, 6])
        )
        assert np.array_equal(
            col_idx,
            np.array(
                [
                    30,
                    2,
                    16,
                    32,
                    31,
                    5,
                    34,
                    6,
                    15,
                    19,
                    10,
                    3,
                    21,
                    8,
                    39,
                    12,
                    24,
                    1,
                    7,
                    35,
                    26,
                    13,
                    22,
                    0,
                    27,
                    17,
                ]
            ),
        )
        print_success_message("test_bootstrapping")

    def test_adaboost(self):
        test_seed = 1
        num_feats = 40
        max_feats = 0.65
        rf_test = RandomForest(4, 5, max_feats)

        np.random.seed(test_seed)
        X_train = np.random.rand(100, num_feats)
        y_train = np.random.randint(0, 2, 100)

        rf_test.adaboost(X_train, y_train)

        assert (
            len(rf_test.alphas) == 4
        ), "The number of alphas should match the number of estimators."

        predictions = rf_test.predict_adaboost(X_train)
        assert predictions.shape == (
            100,
        ), "The shape of predictions should match the number of samples."

        print_success_message("test_adaboost")

    def test_hyperparameter_grid_search(self):
        max_feats = 0.65
        rf_test = RandomForest(4, 5, max_feats)

        permutations = rf_test.hyperparameter_grid_search(
            n_estimators_range=(8, 10, 1),
            max_depth_range=(8, 10, 1),
            max_features_range=(0.7, 1.0, 0.1),
        )

        expected_permutations = [
            (8, 8, 0.7),
            (8, 8, 0.8),
            (8, 8, 0.9),
            (8, 8, 1.0),
            (8, 9, 0.7),
            (8, 9, 0.8),
            (8, 9, 0.9),
            (8, 9, 1.0),
            (8, 10, 0.7),
            (8, 10, 0.8),
            (8, 10, 0.9),
            (8, 10, 1.0),
            (9, 8, 0.7),
            (9, 8, 0.8),
            (9, 8, 0.9),
            (9, 8, 1.0),
            (9, 9, 0.7),
            (9, 9, 0.8),
            (9, 9, 0.9),
            (9, 9, 1.0),
            (9, 10, 0.7),
            (9, 10, 0.8),
            (9, 10, 0.9),
            (9, 10, 1.0),
            (10, 8, 0.7),
            (10, 8, 0.8),
            (10, 8, 0.9),
            (10, 8, 1.0),
            (10, 9, 0.7),
            (10, 9, 0.8),
            (10, 9, 0.9),
            (10, 9, 1.0),
            (10, 10, 0.7),
            (10, 10, 0.8),
            (10, 10, 0.9),
            (10, 10, 1.0),
        ]

        assert len(permutations) == len(
            expected_permutations
        ), "Incorrect number of hyperparameter combinations."

        assert set(permutations) == set(
            expected_permutations
        ), "Grid search output does not match expected values."

        print_success_message("test_hyperparameter_search")


class TestRNN(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.model = RNN(vocab_size=34, max_input_len=40)

    def test_rnn_architecture(self):
        self.model.set_hyperparameters()
        self.model.define_model()
        expected_model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(
                    input_dim=self.model.vocab_size,
                    output_dim=self.model.hp["embedding_dim"],
                ),
                tf.keras.layers.SimpleRNN(units=self.model.hp["rnn_units"]),
                tf.keras.layers.Dense(self.model.vocab_size),
                tf.keras.layers.Activation("softmax"),
            ]
        )

        actual_layers = [layer.get_config() for layer in self.model.model.layers]
        expected_layers = [layer.get_config() for layer in expected_model.layers]

        passed, feedback = assert_network_layers_match(actual_layers, expected_layers)
        if passed:
            print_success_message("test_rnn_architecture")
        else:
            print_fail_message("test_rnn_architecture", feedback)


class TestLSTM(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName)
        self.model = LSTM(vocab_size=34, max_input_len=40)

    def test_lstm_architecture(self):
        self.model.set_hyperparameters()
        self.model.define_model()
        expected_model = tf.keras.Sequential(
            [
                tf.keras.layers.Embedding(
                    input_dim=self.model.vocab_size,
                    output_dim=self.model.hp["embedding_dim"],
                ),
                tf.keras.layers.LSTM(units=self.model.hp["lstm_units"]),
                tf.keras.layers.Dense(self.model.vocab_size),
                tf.keras.layers.Activation("softmax"),
            ]
        )

        actual_layers = [layer.get_config() for layer in self.model.model.layers]
        expected_layers = [layer.get_config() for layer in expected_model.layers]
        passed, feedback = assert_network_layers_match(actual_layers, expected_layers)
        if passed:
            print_success_message("test_lstm_architecture")
        else:
            print_fail_message("test_lstm_architecture", feedback)


def assert_network_layers_match(actual_layers, expected_layers):
    # compare number of layers
    if len(actual_layers) != len(expected_layers):
        feedback = (
            f"\nLayer count mismatch:"
            f"\nActual number of layers: {len(actual_layers)}"
            f"\nExpected number of layers: {len(expected_layers)}"
        )

        return False, feedback

    for i, (actual, expected) in enumerate(zip(actual_layers, expected_layers)):
        actual_config = actual.copy()
        expected_config = expected.copy()

        # Remove name fields because they numbered (e.g lstm_20, lstm_21) based on how many times you've built the model
        actual_layer_name = actual_config.pop("name", None)
        expected_layer_name = expected_config.pop("name", None)

        # extracts base type from the layer name (e.g. lstm_20 -> lstm)
        actual_layer_type = get_layer_type(actual_layer_name)
        expected_layer_type = get_layer_type(expected_layer_name)

        # compare layer types
        if actual_layer_type != expected_layer_type:
            type_diff = (
                f"      Layer type found:   {actual_layer_type}\n"
                f"      Expected layer type: {expected_layer_type}"
            )

            feedback = f"\nMismatch in layer {i}" f"\n{type_diff}"
            return False, feedback

        # compare properties of layer
        if actual_config != expected_config:
            # Find and format differences
            diffs = []
            all_keys = set(actual_config.keys()) | set(expected_config.keys())
            for key in sorted(all_keys):
                actual_value = actual_config.get(key, "<MISSING>")
                expected_value = expected_config.get(key, "<MISSING>")

                if actual_value != expected_value:
                    diffs.append(
                        f"    {key}:\n"
                        f"      Actual:   {actual_value}\n"
                        f"      Expected: {expected_value}"
                    )

            feedback = (
                f"\nMismatch in layer {i}: {actual_layer_type}"
                f"\n\nDifferences found:"
                f"\n{''.join(diffs)}"
            )

            return False, feedback

    return True, None


def get_layer_type(layer_name):
    split = layer_name.split("_")
    if len(split) > 0:
        return split[0]
    return layer_name


def print_array(array):
    print(np.array2string(array, separator=", "))


def print_dict_arrays(dict_arrays):
    for key in dict_arrays:
        print(key)
        print_array(dict_arrays[key])


def print_success_message(test_name):
    print(test_name + " passed!")


def print_fail_message(test_name, feedback=None):
    print(test_name + " failed")
    print(feedback)
