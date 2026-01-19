import re
import logging
import sys
import random
from typing import Mapping, Sequence, Tuple, Literal, Iterator

import torch
import numpy as np
import seaborn as sns
import pandas as pd

from pathlib import Path
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.manifold import TSNE
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_curve,
    auc,
    average_precision_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from utils.gait_parameters_extractor_raw_2d import (
    GaitParametersExtractorRaw2D,
    CoordinatesIdx2D,
)
from utils.torch_siamese_raw import (
    SiameseGaitDatasetRaw,
    SiameseNetworkLSTM,
    SiameseNetworkConv1D,
    ContrastiveLoss,
    compute_similarity,
)


class CrossValidationSiameseRaw2D:
    def __init__(
        self,
        sequence_cycles: Mapping[str, Mapping[str, Sequence]],
        selected_joints: Mapping[str, str],
        coordinates_idx: CoordinatesIdx2D = CoordinatesIdx2D(),
        logger_name: str = "logger",
        log_level: int = logging.DEBUG,
    ):
        self.raw_sequence_cycles = sequence_cycles
        self._logger = self._get_logger(log_level=log_level, name=logger_name)
        participants, cycles_features, features_number = self.prepare_cycles_and_participant_labels(
            sequence_cycles=sequence_cycles,
            selected_joints=selected_joints,
            coordinates_idx=coordinates_idx,
        )
        self.participants = participants
        self.cycles_features = cycles_features
        self.features_number = features_number

    def prepare_cycles_and_participant_labels(
        self,
        sequence_cycles: Mapping[str, Mapping[str, Sequence]],
        selected_joints: Mapping[str, str],
        coordinates_idx: CoordinatesIdx2D = CoordinatesIdx2D(),
    ) -> Tuple[Sequence[int], Sequence[np.ndarray], int]:

        self._logger.info("Preparing cycles and participant labels...")

        pattern = r"p(\d{1,2})s(\d{1,2})c(\d{1,2})"
        combined_participants = []
        combined_sequences_parameters = []

        for sequence_key, sequence_joint_positions in sequence_cycles.items():
            gpe_raw = GaitParametersExtractorRaw2D(
                sequence_parameters = sequence_joint_positions,
                selected_joints = selected_joints,
                coordinates_idx=coordinates_idx,
            )
            sequence_parameters = gpe_raw.get_gait_parameters()
            match = re.search(pattern, sequence_key)
            participant, _, _ = match.groups()

            combined_participants.append(int(participant))
            combined_sequences_parameters.append(sequence_parameters)

        parameters_number = len(gpe_raw.get_gait_parameters_names())

        return combined_participants, combined_sequences_parameters, parameters_number

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
            logging.captureWarnings(True)

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

    @staticmethod
    def _plot_pr_and_roc_curve(
        y_true_values: Sequence[float],
        y_pred_values: Sequence[float],
        show_plot: bool = True,
    ) -> float:
        """
        Function to create two subplots:
         -  Precision-Recall curve
         -  ROC curve with AUROC

        Returns calculated Area Under the ROC Curve (AUROC) value.
        """

        precision, recall, _ = precision_recall_curve(y_true_values, y_pred_values)
        avg_precision = average_precision_score(y_true_values, y_pred_values)

        fpr, tpr, _ = roc_curve(y_true_values, y_pred_values)
        roc_auc = auc(fpr, tpr)

        if show_plot:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))

            # precision-recall plot
            axes[0].plot(recall, precision)
            axes[0].set_xlabel("Recall")
            axes[0].set_ylabel("Precision")
            axes[0].set_title(f"Precision-Recall Curve (AP = {avg_precision:.3f})")
            axes[0].grid(True)

            # ROC plot
            axes[1].plot(fpr, tpr, label=f"AUROC = {roc_auc:.3f}")
            axes[1].plot([0, 1], [0, 1], linestyle="--")
            axes[1].set_xlabel("False Positive Rate")
            axes[1].set_ylabel("True Positive Rate")
            axes[1].set_title("ROC Curve")
            axes[1].legend(loc="lower right")
            axes[1].grid(True)

            plt.tight_layout()
            plt.show()

        return roc_auc

    @staticmethod
    def _prepare_tsne_plot(
        labels: Sequence[int],
        embeddings: Sequence[Sequence[float]],
        perplexity: int = 10,
    ):
        labels = np.array(labels)
        embeddings = np.array(embeddings)

        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(embeddings)

        plt.figure(figsize=(8, 6))

        for ptcp in set(labels):
            plt.scatter(
                X_tsne[labels == ptcp, 0],
                X_tsne[labels == ptcp, 1],
                label=f"Participant {ptcp}",
                alpha=0.7,
                s=30,
            )

        plt.legend()
        plt.title("t-SNE Visualization")
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.tight_layout()
        plt.show()

    def _single_train_iteration(
        self,
        train_dataset: SiameseGaitDatasetRaw,
        siamese_nn_type: Literal["conv1d", "lstm"] = "conv1d",
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        n_epochs: int = 10,
        embedding_size: int = 10,
    ) -> torch.nn.Module:

        if siamese_nn_type == "lstm":
            model = SiameseNetworkLSTM(input_size=self.features_number, embedding_size=embedding_size)
        else:
            model = SiameseNetworkConv1D(input_size=self.features_number, embedding_size=embedding_size)

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
    ):
        y_true_values = []
        y_pred_values = []

        for ptcpt_1, ptcpt_2, label in test_dataset.data:
            y_true_values.append(label.item())
            y_pred_values.append(compute_similarity(ptcpt_1, ptcpt_2, model).item())

        return y_true_values, y_pred_values

    def _single_train_evaluation_with_rank_classification(
        self,
        model: torch.nn.Module,
        test_dataset: SiameseGaitDatasetRaw,
    ):
        y_true_values = []
        y_pred_values = []

        for ptcpt_1, ptcpt_2, label in test_dataset.data:
            y_true_values.append(label.item())
            y_pred_values.append(compute_similarity(ptcpt_1, ptcpt_2, model).item())

        return y_true_values, y_pred_values

    def _calculate_evaluation_metrics(
        self,
        y_true_values: Sequence[float],
        y_pred_values: Sequence[float],
        threshold: float = 0.5,
        show_plot: bool = True,
    ) -> Tuple[float, float, float]:

        y_true = [val == 0 for val in y_true_values]
        y_pred = [val < threshold for val in y_pred_values]

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
        siamese_nn_type: Literal["conv1d", "lstm"] = "conv1d",
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        n_epochs: int = 10,
        threshold: float = 0.5,
        embedding_size: int = 10,
        show_plot: bool = True,
        csv_file_path: Path | str | None = None,
    ):
        cummulated_y_true_values = []
        cummulated_y_pred_values = []

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
                feature_dim=self.features_number,
            )

            test_dataset = SiameseGaitDatasetRaw(
                selected_participants=test_participants,
                all_participants=self.participants,
                features=self.cycles_features,
                feature_dim=self.features_number,
            )

            self._logger.info("Train dataset size: %r", len(train_dataset))
            self._logger.info("Test dataset size: %r", len(test_dataset))

            trained_model = self._single_train_iteration(
                train_dataset=train_dataset,
                siamese_nn_type=siamese_nn_type,
                learning_rate=learning_rate,
                batch_size=batch_size,
                n_epochs=n_epochs,
                embedding_size=embedding_size,
            )

            y_true_values, y_pred_values = self._single_train_evaluation(
                model=trained_model,
                test_dataset=test_dataset,
            )

            self._calculate_evaluation_metrics(
                y_true_values=y_true_values,
                y_pred_values=y_pred_values,
                show_plot=show_plot,
                threshold=threshold,
            )

            cummulated_y_true_values += y_true_values
            cummulated_y_pred_values += y_pred_values

        if csv_file_path is not None:
            self._logger.info("Results will be saved to %r", csv_file_path)
            pd.DataFrame(
                {
                    "true": cummulated_y_true_values,
                    "predicted": cummulated_y_pred_values,
                }
            ).to_csv(csv_file_path)

        self._logger.info("\n")
        self._logger.info("Combined evaluation")
        self._logger.info("%s", "*" * 50)

        auroc = self._plot_pr_and_roc_curve(
            y_pred_values=[min(value, 1) for value in cummulated_y_pred_values],
            y_true_values=cummulated_y_true_values,
            show_plot=show_plot,
        )

        self._logger.info("Area under ROC curve: %r", auroc)

        accuracy, precision, recall = self._calculate_evaluation_metrics(
            y_pred_values=cummulated_y_pred_values,
            y_true_values=cummulated_y_true_values,
            show_plot=show_plot,
            threshold=threshold,
        )

        self._logger.info("& accuracy & precision &  recall  &   auroc  ")
        self._logger.info(
            "& %.2f    & %.2f     & %.2f     & %.2f",
            accuracy * 100,
            precision * 100,
            recall * 100,
            auroc * 100,
        )

        return accuracy, precision, recall, auroc

    def perform_rank_classification_cv(
        self, X: Sequence[Sequence[float]], y: Sequence[int], n_splits: int = 5
    ) -> Mapping[str, float]:
        X = np.array(X)
        y = np.array(y)

        models = {
            "k-NN eucl": KNeighborsClassifier(n_neighbors=5, metric="euclidean"),
            "k-NN manh": KNeighborsClassifier(n_neighbors=5, metric="manhattan"),
            "SVM": SVC(probability=True, kernel="rbf"),
            "MLP": MLPClassifier(
                hidden_layer_sizes=(64, 32), max_iter=500, random_state=42
            ),
            "NB": GaussianNB(),
            "LR": LogisticRegression(),
        }

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        results = {}

        self._logger.info("Starting cross-validated model evaluation...")
        self._logger.info(
            "%-12s | %-10s | %-10s | %-10s",
            "Model",
            "Rank-1",
            "Rank-2",
            "Rank-3",
        )
        self._logger.info("-" * 55)

        for name, clf in models.items():
            r1_scores, r2_scores, r3_scores = [], [], []

            for train_idx, test_idx in skf.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]

                # Scale per fold (important!)
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                clf.fit(X_train, y_train)
                probs = clf.predict_proba(X_test)

                max_k = len(clf.classes_)
                r1_scores.append(
                    top_k_accuracy_score(y_test, probs, k=1, labels=clf.classes_)
                )
                r2_scores.append(
                    top_k_accuracy_score(
                        y_test, probs, k=min(2, max_k), labels=clf.classes_
                    )
                )
                r3_scores.append(
                    top_k_accuracy_score(
                        y_test, probs, k=min(3, max_k), labels=clf.classes_
                    )
                )

            r1 = np.mean(r1_scores)
            r2 = np.mean(r2_scores)
            r3 = np.mean(r3_scores)

            results[name] = [r1, r2, r3]
            self._logger.info("%-12s | %.4f     | %.4f     | %.4f", name, r1, r2, r3)

        return results

    def perform_training_with_rank_classification(
        self,
        n_folds: int = 5,
        siamese_nn_type: Literal["conv1d", "lstm"] = "conv1d",
        learning_rate: float = 1e-4,
        batch_size: int = 32,
        n_epochs: int = 10,
        embedding_size: int = 10,
        tsne_perplexity: int = 10,
        show_plot: bool = True,
        csv_file_path: Path | str | None = None,
    ):

        iteration = 1

        train_test_folds_iterator = self._prepare_train_test_folds(n_folds)

        rank_classification_results = []
        test_sets_sizes = []

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

            self._logger.info("Train dataset size: %r", len(train_dataset))

            trained_model = self._single_train_iteration(
                train_dataset=train_dataset,
                siamese_nn_type=siamese_nn_type,
                learning_rate=learning_rate,
                batch_size=batch_size,
                n_epochs=n_epochs,
                embedding_size=embedding_size,
            )

            test_labels = []
            test_participants_embeddings = []

            for test_participant in test_participants:
                for idx, tmp_ptcp in enumerate(self.participants):
                    if tmp_ptcp == test_participant:
                        test_labels.append(test_participant)
                        ptcp_features = self.cycles_features[idx]
                        ptcp_features_tensor = (
                            torch.from_numpy(ptcp_features.T).float().unsqueeze(0)
                        )
                        ptcp_features_embedding = (
                            trained_model.forward_once(ptcp_features_tensor)
                            .detach()
                            .numpy()[0]
                        )
                        test_participants_embeddings.append(ptcp_features_embedding)

            for test_label, embedding in zip(test_labels, test_participants_embeddings):
                self._logger.debug(
                    "Participant: %r Embedding: %r", test_label, list(embedding)
                )

            if show_plot:
                self._prepare_tsne_plot(
                    labels=test_labels,
                    embeddings=test_participants_embeddings,
                    perplexity=tsne_perplexity,
                )

            self._logger.info(
                "Test set size for rank classification: %r", len(test_labels)
            )

            results = self.perform_rank_classification_cv(
                X=test_participants_embeddings, y=test_labels
            )

            rank_classification_results.append(results)
            test_sets_sizes.append(len(test_labels))

        combined_results = {
            clf_name: [0, 0, 0]
            for clf_name in [
                "k-NN eucl",
                "k-NN manh",
                "SVM",
                "MLP",
                "NB",
                "LR",
            ]
        }

        for clsf_results, set_size in zip(rank_classification_results, test_sets_sizes):
            for clf_name, accuracies in clsf_results.items():
                for i, accuracy in enumerate(accuracies):
                    combined_results[clf_name][i] += accuracy * set_size

        self._logger.info("\n")
        self._logger.info(
            "Combined results from all folds (calculated with weighted average)"
        )
        self._logger.info("%s", "*" * 50)
        self._logger.info(
            "%-10s & %-10s & %-10s & %-10s",
            "Model",
            "Rank-1",
            "Rank-2",
            "Rank-3",
        )
        self._logger.info("%s", "-" * 50)

        for clf_name, combined_accuracies in combined_results.items():
            avg_weighted_accuracies = [
                100 * acc / sum(test_sets_sizes) for acc in combined_accuracies
            ]
            # converted to % and format easier to use in overleaf
            self._logger.info(
                "%-10s & %.2f      & %.2f      & %.2f",
                clf_name,
                avg_weighted_accuracies[0],
                avg_weighted_accuracies[1],
                avg_weighted_accuracies[2],
            )

        self._logger.info("*" * 50)
