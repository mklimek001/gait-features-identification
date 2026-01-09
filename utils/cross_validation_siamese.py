import logging
import sys
from typing import Sequence, Iterable, TypeVar, Tuple, Literal

import torch
import numpy as np
import pandas as pd
import seaborn as sns
from collections import Counter
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
)

from utils.torch_siamese import (
    SiameseGaitDataset,
    SiameseNetwork,
    ContrastiveLoss,
    compute_similarity,
    SiameseNetworkCosine,
    ContrastiveLossCosine,
    compute_similarity_cosine,
)


T = TypeVar("T")


class CVTrainer:
    """
    Class to perform cross-validation of siamese neural network training
    for person identification based on gait features extracted from 3D pose position.
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        selected_features: Sequence[str] | None = None,
        splits_number: int = 5,
        log_level: int = logging.DEBUG,
    ):
        self._logger = self._get_logger(log_level=log_level)
        self.original_dataset: pd.DataFrame = dataset
        if selected_features is None:
            self._logger.info(
                "Provided selected_features parameter is None - all features from dataset will be used."
            )
        self.selected_features: Sequence[str] = (
            [col_name for col_name in dataset.columns if col_name != "participant"]
            if selected_features is None
            else selected_features
        )
        self._logger.info(
            "Number of selected features: %d", len(self.selected_features)
        )

        self.original_participants: Sequence[int] = dataset["participant"]
        self.features_df: pd.DataFrame = dataset.drop(columns=["participant"])[
            self.selected_features
        ]
        self.scaled_features_df = self._scale_dataframe(dataframe=self.features_df)
        self.splits_numbers: int = splits_number
        self.splits: Sequence[Tuple[Sequence[str], Sequence[str]]] = (
            self._determine_fold_subdatasets()
        )

    def _get_logger(self, log_level: int, name: str = "logger") -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(log_level)

        # Prevent adding multiple handlers if this method is called repeatedly
        if not logger.handlers:
            handler = logging.StreamHandler(sys.stdout)

            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)

            logger.addHandler(handler)

        return logger

    def _scale_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(dataframe)
        return pd.DataFrame(features_scaled)

    def _determine_fold_subdatasets(
        self,
    ) -> Sequence[Tuple[Sequence[str], Sequence[str]]]:
        """
        Method to find participants test subsets with following algorithm:

        1. Determine if participant has been recorded once or twice (4 or 8 samples in dataset)
        2. Participants recorded twice are used only for training for bigger size of dataset
        3. Participants recorded once are randomly divided into `n` subsets
        4. For created subsets list of tuples with (train_split, test_split) is created and returned
        """
        two_walks_ptcpts, one_walk_ptcpts = [], []
        for ptcpt, count in self.original_participants.value_counts().to_dict().items():
            if count == 8:
                two_walks_ptcpts.append(ptcpt)
            else:
                one_walk_ptcpts.append(ptcpt)

        self._logger.info("Found %s participants recorded once.", len(one_walk_ptcpts))
        self._logger.debug("Participants recorded once: %s", one_walk_ptcpts)
        self._logger.info(
            "Found %s participants recorded twice.", len(two_walks_ptcpts)
        )
        self._logger.debug("Participants recorded twice: %s", two_walks_ptcpts)

        test_splits = self.__split_into_random_sublists(
            one_walk_ptcpts, self.splits_numbers
        )
        train_test_splits = [
            (
                self._subtract_two_sequences(one_walk_ptcpts, test_split)
                + two_walks_ptcpts,
                test_split,
            )
            for test_split in test_splits
        ]

        return train_test_splits

    @staticmethod
    def __split_into_random_sublists(ptcpts_lst: Sequence, n: int):
        splits = [[] for _ in range(n)]
        for idx, ptcpt in enumerate(ptcpts_lst):
            splits[idx % n].append(ptcpt)
        return splits

    @staticmethod
    def _subtract_two_sequences(
        sequence_1: Iterable[T], sequence_2: Iterable[T]
    ) -> list[T]:
        return list(set(sequence_1) - set(sequence_2))

    def _single_train_iteration(
        self,
        train_participants: Sequence[int],
        test_participants: Sequence[int],
        n_epochs: int = 10,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        embedding_size: int = 10,
        criterion_type: Literal["euclidean", "cosine"] = "euclidean",
    ) -> Tuple[Sequence[bool], Sequence[bool]]:
        """
        Single iteration of siamese neural network training with one fold of train-test splits.
        """

        train_dataset = SiameseGaitDataset(
            selected_participants=train_participants,
            all_participants=self.original_participants,
            features_scaled_df=self.scaled_features_df,
        )

        test_dataset = SiameseGaitDataset(
            selected_participants=test_participants,
            all_participants=self.original_participants,
            features_scaled_df=self.scaled_features_df,
        )

        self._logger.info("Train and test datasets created successfully")
        self._logger.info("Train dataset size: %d", len(train_dataset))
        self._logger.info("Test dataset size: %d", len(test_dataset))

        if criterion_type == "euclidean":
            model = SiameseNetwork(
                input_size=len(self.selected_features), embedding_size=embedding_size
            )
            criterion = ContrastiveLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            model = SiameseNetworkCosine(
                input_size=len(self.selected_features), embedding_size=embedding_size
            )
            criterion = ContrastiveLossCosine()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        self._logger.info("Siamese neural network model training started...")

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
                "Epoch %d, Train Loss: %.4f", epoch + 1, train_loss / len(train_loader)
            )

        self._logger.info("Siamese neural network model training finished")

        y_true = []
        y_pred = []
        threshold = 0.5

        for ptcpt_1, ptcpt_2, label in test_dataset.data:
            y_true.append(label.item() == 0)
            if criterion_type == "euclidean":
                y_pred.append(
                    compute_similarity(ptcpt_1, ptcpt_2, model).item() < threshold
                )
            else:
                y_pred.append(
                    compute_similarity_cosine(ptcpt_1, ptcpt_2, model).item()
                    < threshold
                )

        return y_true, y_pred

    def _calculate_confusion_matrix(
        self, y_true: Sequence[bool], y_pred: Sequence[bool]
    ) -> pd.DataFrame:
        """
        Prepare models confusion matrix based on provided results.
        """
        cm = confusion_matrix(y_true, y_pred, labels=[True, False])
        class_names = ["True", "False"]
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        self._logger.info("Confusion matrix: \n %s", cm_df)
        return cm_df

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

    def _calculate_base_metrics(
        self, y_true: Sequence[bool], y_pred: Sequence[bool]
    ) -> tuple[float, float, float]:
        """
        Calculate models accuracy, precision and recall based on provided results.
        """
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        self._logger.info("Accuracy: %.3f", accuracy)
        self._logger.info("Precision: %.3f", precision)
        self._logger.info("Recall: %.3f", recall)

        return accuracy, precision, recall

    def run_training(
        self,
        n_epochs: int = 10,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        embedding_size: int = 10,
        show_plot=True,
        criterion_type: Literal["euclidean", "cosine"] = "euclidean",
    ) -> tuple[float, float, float]:
        """
        Run train loop for all folds.
        Training will include n_epochs epochs with provided learning rate'
        """

        cumulated_y_true, cumulated_y_pred = [], []

        for index, (train, test) in enumerate(self.splits):
            self._logger.info(
                "Training iteration %d/%d", index + 1, self.splits_numbers
            )
            y_true, y_pred = self._single_train_iteration(
                train_participants=train,
                test_participants=test,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                embedding_size=embedding_size,
                criterion_type=criterion_type,
            )
            cumulated_y_true += y_true
            cumulated_y_pred += y_pred

            self._logger.info("Iteration results: ")
            self._calculate_confusion_matrix(y_true, y_pred)
            self._calculate_base_metrics(y_true, y_pred)

        self._logger.info("Training loop for all folds finished.")
        self._logger.info("Final results: ")
        cm_df = self._calculate_confusion_matrix(cumulated_y_true, cumulated_y_pred)
        accuracy, precision, recall = self._calculate_base_metrics(
            cumulated_y_true, cumulated_y_pred
        )
        if show_plot:
            self._prepare_confusion_matrix_plot(cm_df=cm_df)
        return accuracy, precision, recall

    def _single_train_iteration_rank_classification(
        self,
        train_participants: Sequence[int],
        test_participants: Sequence[int],
        n_epochs: int = 10,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        embedding_size: int = 10,
        criterion_type: Literal["euclidean", "cosine"] = "euclidean",
        k_neighbors: int = 5,
    ) -> Tuple[Sequence[bool], Sequence[bool]]:
        """
        Single iteration of siamese neural network training with one fold of train-test splits.
        Adjusted for rank-k classification with kNN.
        """

        train_dataset = SiameseGaitDataset(
            selected_participants=train_participants,
            all_participants=self.original_participants,
            features_scaled_df=self.scaled_features_df,
        )

        test_dataset = SiameseGaitDataset(
            selected_participants=test_participants,
            all_participants=self.original_participants,
            features_scaled_df=self.scaled_features_df,
        )

        self._logger.info("Train and test datasets created successfully")
        self._logger.info("Train dataset size: %d", len(train_dataset))
        self._logger.info("Test dataset size: %d", len(test_dataset))

        if criterion_type == "euclidean":
            model = SiameseNetwork(
                input_size=len(self.selected_features), embedding_size=embedding_size
            )
            criterion = ContrastiveLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        else:
            model = SiameseNetworkCosine(
                input_size=len(self.selected_features), embedding_size=embedding_size
            )
            criterion = ContrastiveLossCosine()
            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        self._logger.info("Siamese neural network model training started...")

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
                "Epoch %d, Train Loss: %.4f", epoch + 1, train_loss / len(train_loader)
            )

        self._logger.info("Siamese neural network model training finished")

        knn_scaled_features = pd.DataFrame()
        knn_participants_labels = []

        for participant in test_participants:
            self._logger.debug("participant: %r", participant)
            mask = self.original_participants.isin([participant])
            participant_scaled_features = self.scaled_features_df[mask].reset_index(
                drop=True
            )
            knn_scaled_features = pd.concat(
                [participant_scaled_features, knn_scaled_features]
            )
            knn_participants_labels += [
                participant for _ in range(len(participant_scaled_features))
            ]

        knn_scaled_features.reset_index(inplace=True, drop=True)
        self._logger.debug("Test scaled features = \n%r", knn_scaled_features)
        self._logger.info("Test scaled features shape = %r", knn_scaled_features.shape)
        self._logger.debug("Test participants labels: %r", knn_participants_labels)
        assert knn_scaled_features.shape[0] == len(knn_participants_labels)

        knn_embeddings = []

        for _, row in knn_scaled_features.iterrows():
            self._logger.debug("Row: %r", row.to_list())
            row_embedding = model.forward_once(torch.tensor(row.to_list())).tolist()
            self._logger.debug("Row embedding: %r", row_embedding)
            knn_embeddings.append(row_embedding)

        self._logger.info(
            "Obtained embeddings shape = %r",
            (len(knn_embeddings), len(knn_embeddings[0])),
        )

        X_train, X_test, y_train, y_test = train_test_split(
            np.array(knn_embeddings),
            np.array(knn_participants_labels),
            test_size=0.4,
            stratify=np.array(knn_participants_labels),
            random_state=42,
        )

        knn = NearestNeighbors(n_neighbors=k_neighbors, metric="euclidean")
        knn.fit(X_train)

        distances, indices = knn.kneighbors(X_test)

        rank1_correct = 0
        rank5_correct = 0

        for i, neigh_idx in enumerate(indices):
            neighbor_labels = y_train[neigh_idx]
            self._logger.debug("Neighbor labels: %r", neighbor_labels)

            pred_rank1 = Counter(neighbor_labels).most_common(1)[0][0]
            if pred_rank1 == y_test[i]:
                rank1_correct += 1

            if y_test[i] in neighbor_labels:
                rank5_correct += 1

        rank1_acc = rank1_correct / len(y_test)
        rank5_acc = rank5_correct / len(y_test)

        self._logger.info(f"Rank-1 Accuracy: {rank1_acc:.3f}")
        self._logger.info(f"Rank-5 Accuracy: {rank5_acc:.3f}")

        return [], []

    def run_training_rank_classification(
        self,
        n_epochs: int = 10,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        embedding_size: int = 10,
        show_plot=True,
        criterion_type: Literal["euclidean", "cosine"] = "euclidean",
        k_neighbors: int = 3,
    ) -> tuple[float, float, float]:
        """
        Run train loop for all folds.
        Training will include n_epochs epochs with provided learning rate'
        """

        cumulated_y_true, cumulated_y_pred = [], []

        for index, (train, test) in enumerate(self.splits):
            self._logger.info(
                "Training iteration %d/%d", index + 1, self.splits_numbers
            )
            y_true, y_pred = self._single_train_iteration_rank_classification(
                train_participants=train,
                test_participants=test,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                batch_size=batch_size,
                embedding_size=embedding_size,
                criterion_type=criterion_type,
                k_neighbors=k_neighbors,
            )
            cumulated_y_true += y_true
            cumulated_y_pred += y_pred

            self._logger.info("Iteration results: ")
            self._calculate_confusion_matrix(y_true, y_pred)
            self._calculate_base_metrics(y_true, y_pred)

        self._logger.info("Training loop for all folds finished.")
        self._logger.info("Final results: ")
        cm_df = self._calculate_confusion_matrix(cumulated_y_true, cumulated_y_pred)
        accuracy, precision, recall = self._calculate_base_metrics(
            cumulated_y_true, cumulated_y_pred
        )
        if show_plot:
            self._prepare_confusion_matrix_plot(cm_df=cm_df)
        return accuracy, precision, recall


if __name__ == "__main__":
    gait_df = pd.read_csv("./datasets/gait_features_2.csv")
    cv_trainer = CVTrainer(gait_df)
    cv_trainer.run_training()
