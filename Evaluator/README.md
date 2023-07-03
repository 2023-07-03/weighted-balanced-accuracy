# WBA-Evaluator

Class distribution skews in imbalanced datasets may lead to models with prediction bias towards majority classes, making fair assessment of classifiers a challenging task. Balanced Accuracy is a popular metric used to evaluate a classifier’s prediction performance under such scenarios. However, this metric falls short when classes vary in importance, especially when class importance is skewed differently from class cardinality distributions.  In this paper, we study this problem within the context of several real-world classification use cases, and propose a simple and general-purpose evaluation framework for imbalanced data classification that is sensitive to arbitrary skews in class cardinalities and importances.  Experiments with several state-of-the-art classifiers tested on real-world datasets from public and industrial benchmarks show that our new framework can evaluate and rank results more effectively than its existing counterparts.

# Running
The way to run WBA-Evaluator is as below:
```
python wba-evaluator.py {-c} <real_data_file> <predicted_data_file> {-w} <user_defined_weight> {-v} <mode>
```
<br/>
-v : Produce verbose output <br/>
-c : Compute by providing class distribution file and misclassification by model file <br/>
real_data_file : File with real data labels <br/>
predicted_data_file : File with predicted data labels <br/>
mode : Weight calculation method for WBA. It can be either 0 for balanced weights (BA), 1 for user-defined weights, 2 for rarity weights, 3 for composite of rarity and user-defined weights, and 4 for composite weights (any n criterias). Default = 1. <br/>
-w : Option used for user defined weights for the classes. <br/>
user_defined_weight : File with weights of classes. Used with option -w <br/>

To use the composition mode with any n kinds of weights (i.e., mode 4), give multiple filenames containing the weight of classes.

See scripts folder to see how examples are run.

