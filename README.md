# **Deep Learning N-Days Daily Roadmap**


## **Phase 1: Foundations (Day 1–8)**

| Day | Topic                     | Goal                                                                                                     |
| --- | ------------------------- | -------------------------------------------------------------------------------------------------------- |
| 1   | What is Deep Learning?    | Introduction, history, real-world examples, first neuron intuition, forward propagation of single neuron |
| 2   | Neurons, Weights, Bias & Activations    | Deep dive into neuron structure, weights, bias, activation functions, visualizations       |
| 3   | Forward Propagation       | Manual calculations, numpy implementation, ReLU, Sigmoid, Tanh                                           |
| 4   | Tiny Neural Network       | Build 2-layer network from scratch in numpy                                                              |
| 5   | Loss Functions            | MSE, Cross-Entropy, small examples                                                                       |
| 6   | Optimizers                | Gradient Descent intuition, learning rate, update rules, simple code                                     |
| 7   | Mini Exercise             | Manual neuron calculations, small network experiments                                                    |
| 8   | Phase Summary | wrap-up, code + visuals                                             |


## **Phase 2: First Neural Networks & Backpropagation (Day 9–19)**

| Day | Topic                           | Goal                                    |
| --- | ------------------------------- | -------------------------------------------------------- |
| 9   | Gradient & Derivative Intuition | Calculus review, small examples                          |
| 10  | Backpropagation Basics          | Manual chain rule calculations                           |
| 11  | Backpropagation in Numpy        | Tiny network training step-by-step                       |
| 12  | Full Forward + Backprop Example | Numpy mini training loop                                 |
| 13  | PyTorch Introduction            | Tensors, basic operations, GPU usage                     |
| 14  | PyTorch Neural Network          | Build first simple model                                 |
| 15  | Training loop in PyTorch        | Forward + loss + backward + optimizer step               |
| 16  | Overfitting vs Underfitting     | Visualize loss curves, concept explanation               |
| 17  | Validation & Test split         | Data handling in PyTorch, metrics                        |
| 18  | Hyperparameters                 | Learning rate, epochs, batch size, grid search intuition |
| 19  | Phase Summary                   | Notebook wrap-up, code + visuals                         |


## **Phase 3: Convolutional & Recurrent Networks (Day 20–33)**

| Day | Topic                       | Goal                            |
| --- | --------------------------- | ------------------------------------------------- |
| 20  | CNN introduction               | Filters, stride, padding, convolution example     |
| 21  | CNN Layers                  | Pooling, flatten, fully connected layers          |
| 22  | CNN in PyTorch              | Build simple CNN for MNIST                        |
| 23  | CNN training                | Forward + backward + optimizer, visualize filters |
| 24  | RNN introduction               | Sequence data, hidden state, unrolling            |
| 25  | LSTM & GRU                  | Why LSTM > RNN, gates explanation                 |
| 26  | Simple RNN in PyTorch       | Manual sequence prediction                        |
| 27  | LSTM example                | Text sequence prediction                          |
| 28  | NLP preprocessing           | Tokenization, embedding, padding                  |
| 29  | RNN mini project            | Predict sentiment on small dataset                |
| 30  | CNN + RNN comparison        | When to use which, pros/cons                      |
| 31  | Regularization in CNN/RNN   | Dropout, batch norm, visualization                |
| 32  | Hyperparameters for CNN/RNN | Learning rate, optimizer tuning                   |
| 33  | Phase Summary               | Notebook wrap-up with multiple small examples     |


## **Phase 4: CNN Advanced Mastery (Day 34–45)**

| Day | Focus                              | Goal                                                                       |
| --- | ---------------------------------- | -------------------------------------------------------------------------- |
| 34  | CNN Regularization                 | Dropout placement, BatchNorm behavior (train vs eval), overfitting control |
| 35  | CNN Weight Initialization          | Xavier vs He, dead ReLU problem, empirical comparisons                     |
| 36  | CNN Optimizers                     | SGD vs Adam vs AdamW vs RMSProp, convergence tradeoffs                     |
| 37  | CNN Learning Rate Scheduling       | StepLR, ReduceLROnPlateau, CosineAnnealing, LR intuition                   |
| 38  | CNN Data Augmentation              | Geometric & color transforms, over/under-augmentation risks                |
| 39  | CNN Multi-Class Classification     | Softmax, CrossEntropy, class imbalance, top-k accuracy                     |
| 40  | CNN Multi-Label Classification     | BCEWithLogits, threshold tuning, PR tradeoffs                              |
| 41  | CNN Evaluation & Debugging         | Confusion matrix, ROC/PR curves, error analysis                            |
| 42  | CNN Early Stopping & Checkpointing | Validation-driven stopping, best-model saving                              |
| 43  | CNN Hyperparameter Tuning          | Batch size–LR coupling, weight decay, controlled experiments               |
| 44  | CNN Advanced Mini Project          | Apply all CNN techniques end-to-end (no shortcuts)                         |
| 45  | CNN Mastery Validation             | Rebuild CNN from scratch, justify every design decision                    |


## **Phase 5: MLOps / Deployment / Scaling (Day 46–55)**

| Day    | Topic                     | Goal                                                                                          |
| ------ | ------------------------- | --------------------------------------------------------------------------------------------- |
| **46** | Intro to Model Deployment | Understand research vs production, what “serving a model” means, end-to-end pipeline overview |
| **47** | Saving & Loading Models   | Use `torch.save` / `torch.load`, save best CNN model, test loading correctness                |
| **48** | Inference Pipeline        | Build function: image → preprocess → model → prediction (no training code)                    |
| **49** | Flask Basics              | Learn routes, request/response, run a simple API locally                                      |
| **50** | Deploy CNN with Flask     | Wrap your CNN model into API (`/predict` endpoint), test with image input                     |
| **51** | FastAPI Introduction      | Build same API using FastAPI, understand why it's preferred                                   |
| **52** | Docker Basics             | Learn Dockerfile, build image, run container locally                                          |
| **53** | Docker + Model API        | Containerize your FastAPI/Flask app + CNN model                                               |
| **54** | GPU Inference & Batching  | Run batch predictions, measure speed, understand CPU vs GPU inference                         |
| **55** | Logging & Version Control | Add basic logging, use Git properly, intro to DVC (optional)                                  |

## **Phase 6: Object Detection Mastery (Day 56–67)**

| Day    | Topic                                | Goal                                                                                                        |
| ------ | ------------------------------------ | ----------------------------------------------------------------------------------------------------------- |
| **56** | Detection Fundamentals               | Understand classification vs detection, bounding boxes, IoU, NMS, mAP metric                                |
| **57** | Dataset Selection & Preparation      | Choose dataset (COCO subset / Pascal VOC / custom), organize images & annotations, train/test split         |
| **58** | CNN Backbone Review                  | Review your CNN knowledge, understand feature maps, anchors, and why CNNs are backbone of detectors         |
| **59** | Pretrained Detection Models          | Explore YOLOv5 / YOLOv8 inference on sample images, visualize predictions, understand confidence thresholds |
| **60** | Fine-tuning Detection Model          | Load pretrained model, fine-tune on small custom dataset, adjust anchors and learning rate                  |
| **61** | Advanced Training Tricks             | Use augmentation (flips, scale, color), early stopping, learning rate scheduling for detection              |
| **62** | Evaluation & Metrics                 | Compute mAP, IoU; analyze errors, visualize false positives/negatives                                       |
| **63** | Multi-class / Multi-object Scenarios | Handle multiple objects per image, class imbalance, threshold tuning                                        |
| **64** | Optimization & Inference             | Batch inference, GPU utilization, speed-memory tradeoffs, confidence calibration                            |
| **65** | Detection Project                    | Build a mini project: custom dataset, trained model, evaluation, visualizations                             |
| **66** | Deployment                           | Wrap detection model in a simple FastAPI endpoint locally, allow image upload → detection output            |
| **67** | Portfolio Polish                     | Prepare notebook for portfolio: clean code, explanations, results, plots                                    |
