import re
import logging
import sys
import random

import torch
import numpy as np
import seaborn as sns
import pandas as pd

from torch.utils.data import DataLoader
from typing import Mapping, Sequence, Tuple, Literal, Iterator
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)

from utils.gait_parameters_extractor_raw import (
    GaitParametersExtractorRaw,
    CoordinatesIdx,
)
from utils.torch_siamese_raw import (
    SiameseGaitDatasetRaw,
    SiameseNetworkLSTM,
    SiameseNetworkConv1D,
    ContrastiveLoss,
    compute_similarity,
)


class CrossValidationSiameseRaw:
    def __init__(
        self,
        sequence_cycles: Mapping[str, np.ndarray],
        coordinates_idx: CoordinatesIdx = CoordinatesIdx(2, 0, 1),
        log_level: int = logging.DEBUG,
    ):
        self.raw_sequence_cycles = sequence_cycles
        self._logger = self._get_logger(log_level=log_level)
        participants, cycles_features = self.prepare_cycles_and_participant_labels(
            sequence_cycles, coordinates_idx
        )
        self.participants = participants
        self.cycles_features = cycles_features

    def prepare_cycles_and_participant_labels(
        self,
        sequence_cycles: Mapping[str, np.ndarray],
        coordinates_idx: CoordinatesIdx = CoordinatesIdx(2, 0, 1),
    ) -> Tuple[Sequence[int], Sequence[np.ndarray]]:

        self._logger.info("Preparing cycles and participant labels...")

        pattern = r"p(\d{1,2})s(\d{1,2})c(\d{1,2})"
        combined_participants = []
        combined_sequences_parameters = []

        for sequence_key, sequence_joint_positions in sequence_cycles.items():
            gpe_raw = GaitParametersExtractorRaw(
                sequence_joint_positions, coordinates_idx=coordinates_idx
            )
            sequence_parameters = gpe_raw.get_gait_parameters()
            match = re.search(pattern, sequence_key)
            participant, _, _ = match.groups()

            combined_participants.append(int(participant))
            combined_sequences_parameters.append(sequence_parameters)

        return combined_participants, combined_sequences_parameters

    def count_participants_samples(self):
        for i in range(1, 33):
            self._logger.debug(
                "Participant %r -> %r samples", i, self.participants.count(i)
            )

    def _get_logger(self, log_level: int, name: str = "logger") -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(log_level)

        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    @staticmethod
    def _prepare_confusion_matrix_plot(cm_df: pd.DataFrame):
        assert cm_df.shape == (
            2,
            2,
        ), "Provided dataframe is cannot be interpreted as confusion matrix"
        sns.heatmap(
            cm_df,
            annot=True,
            fmt="d",
            cmap=LinearSegmentedColormap.from_list("brownish", ["#f0e5d8", "#54290b"]),
            annot_kws={"size": 14},
        )
        plt.ylabel("Actual", fontsize=12)
        plt.xlabel("Predicted", fontsize=12)
        plt.title("Confusion Matrix", fontsize=16)

        plt.tight_layout()
        plt.savefig("./plots/confusion_matrix_siamese.png")
        plt.show()

    def _single_train_iteration(
        self,
        train_dataset: SiameseGaitDatasetRaw,
        siamese_nn_type: Literal["conv1d", "lstm"] = "lstm",
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        n_epochs: int = 10,
    ) -> torch.nn.Module:

        if siamese_nn_type == "lstm":
            model = SiameseNetworkLSTM()
        else:
            model = SiameseNetworkConv1D()

        criterion = ContrastiveLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        for epoch in range(n_epochs):
            train_dataset.regenerate_pairs()
            train_loader = DataLoader(
                train_dataset, batch_size=batch_size, shuffle=True
            )

            model.train()
            train_loss = 0
            for x1, x2, label in train_loader:
                out1, out2 = model(x1, x2)
                loss = criterion(out1, out2, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            self._logger.info(
                "Epoch %r, Train Loss: %r", epoch + 1, train_loss / len(train_loader)
            )

        model.eval()

        return model

    def _single_train_evaluation(
        self,
        model: torch.nn.Module,
        test_dataset: SiameseGaitDatasetRaw,
        threshold: float = 0.5,
    ):
        y_true = []
        y_pred = []

        for ptcpt_1, ptcpt_2, label in test_dataset.data:
            y_true.append(label.item() == 0)
            y_pred.append(
                compute_similarity(ptcpt_1, ptcpt_2, model).item() < threshold
            )

        return y_true, y_pred

    def _calculate_evaluation_metrics(
        self,
        y_true: Sequence[float],
        y_pred: Sequence[float],
        show_plot: bool = True,
    ) -> Tuple[float, float, float]:

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)

        self._logger.info("Accuracy: %r", accuracy)
        self._logger.info("Precision: %r", precision)
        self._logger.info("Recall: %r", recall)

        cm = confusion_matrix(y_true, y_pred, labels=[True, False])
        class_names = ["True", "False"]
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        self._logger.info("Confusion matrix: \n %r", cm_df)

        if show_plot:
            self._prepare_confusion_matrix_plot(cm_df=cm_df)

        return accuracy, precision, recall

    def _prepare_train_test_folds(
        self,
        folds: int = 5,
    ) -> Iterator[Tuple[Sequence[int], Sequence[int]]]:
        random.seed(42)
        participants = list(range(1, 33))
        random.shuffle(participants)
        test_folds = [[] for _ in range(folds)]
        for idx, item in enumerate(participants):
            test_folds[idx % 5].append(item)

        train_folds = [
            [i for i in range(1, 33) if not i in test_fold] for test_fold in test_folds
        ]
        for i in range(folds):
            self._logger.debug("[%r] Test fold participants: %r", i, test_folds[i])
            self._logger.debug("[%r] Train fold participants: %r ", i, train_folds[i])

        return zip(train_folds, test_folds)

    def perform_training(
        self,
        n_folds: int = 5,
        siamese_nn_type: Literal["conv1d", "lstm"] = "lstm",
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        n_epochs: int = 10,
        threshold: float = 0.5,
        show_plot: bool = True,
    ):
        cummulated_y_true = []
        cummulated_y_pred = []

        iteration = 1

        train_test_folds_iterator = self._prepare_train_test_folds(n_folds)
        for train_participants, test_participants in train_test_folds_iterator:
            self._logger.info("[Iteration %r/%r]", iteration, n_folds)
            iteration += 1

            self._logger.debug("Test fold participants: %r", sorted(test_participants))
            self._logger.debug(
                "Train fold participants: %r", sorted(train_participants)
            )

            train_dataset = SiameseGaitDatasetRaw(
                selected_participants=train_participants,
                all_participants=self.participants,
                features=self.cycles_features,
            )

            test_dataset = SiameseGaitDatasetRaw(
                selected_participants=test_participants,
                all_participants=self.participants,
                features=self.cycles_features,
            )

            self._logger.info("Train dataset size: %r", len(train_dataset))
            self._logger.info("Test dataset size: %r", len(test_dataset))

            trained_model = self._single_train_iteration(
                train_dataset=train_dataset,
                siamese_nn_type=siamese_nn_type,
                learning_rate=learning_rate,
                batch_size=batch_size,
                n_epochs=n_epochs,
            )

            y_true, y_pred = self._single_train_evaluation(
                model=trained_model,
                test_dataset=test_dataset,
                threshold=threshold,
            )

            self._calculate_evaluation_metrics(
                y_true=y_true, y_pred=y_pred, show_plot=show_plot
            )

            cummulated_y_true += y_true
            cummulated_y_pred += y_pred

        self._logger.info("Combined evaluation")
        return self._calculate_evaluation_metrics(
            y_pred=cummulated_y_pred, y_true=cummulated_y_true, show_plot=show_plot
        )
