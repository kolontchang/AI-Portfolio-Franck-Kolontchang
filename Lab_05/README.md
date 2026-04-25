# Lab 05: Sequence Models (RNN, LSTM, GRU)

## Problem Statement
Processing sequential data like text requires networks that hold memory of past inputs. The objective was to build and compare Vanilla RNNs, LSTMs, and GRUs from scratch to classify text from the AG News dataset.

## Approach and Methodology
Implemented text tokenization, vocabulary building, and custom PyTorch `Dataset`/`DataLoader` pipelines. Constructed bidirectional RNN, LSTM, and GRU models. Evaluated each architecture's ability to retain long-term dependencies.

## Results and Evaluation
LSTMs and GRUs significantly outperformed Vanilla RNNs on validation accuracy. The models successfully classified news articles into correct categories by capturing contextual meaning from word sequences.

## Your Learning Outcomes
I mastered the mechanics of gated sequence models and embeddings. By observing the Vanilla RNN struggle, I developed a deep intuition for why LSTMs/GRUs were invented to manage the vanishing gradient problem.

## Requirements or Dependencies
* Ensure `requirements.txt` from the root directory is installed.
* Standard Python 3.8+ environment with PyTorch.

## Sample Data
* Instructions for data access or the necessary subsets are detailed within the respective Jupyter Notebook cells or provided through standard torch/ torchvision datasets.
