echo ""
echo "#######################################"

echo ""
echo "URL Classification vanilla training EXAMPLE - BALANCED ACCURACY IN WBA"
echo ""
python3 ../src/wba-evaluator.py ../examples/URLClassification/url_train_2classes_not-weighted.real ../examples/URLClassification/url_train_2classes_not-weighted.pred -m 0
echo ""
echo "#######################################"

echo ""
echo "URL Classification vanilla training EXAMPLE - INVERSE FREQUENCY WEIGHTS IN WBA"
echo ""
python3 ../src/wba-evaluator.py ../examples/URLClassification/url_train_2classes_not-weighted.real ../examples/URLClassification/url_train_2classes_not-weighted.pred -m 2
echo ""
echo "#######################################"

echo ""
echo "URL Classification vanilla training EXAMPLE - USER DEFINED WEIGHTS IN WBA"
echo ""
python3 ../src/wba-evaluator.py ../examples/URLClassification/url_train_2classes_not-weighted.real ../examples/URLClassification/url_train_2classes_not-weighted.pred -m 1 -w ../examples/URLClassification/url_2classes_user.weights
echo ""
echo "#######################################"

echo ""
echo "URL Classification training with rarity weights EXAMPLE - BALANCED ACCURACY IN WBA"
echo ""
python3 ../src/wba-evaluator.py ../examples/URLClassification/url_train_2classes_weighted-rarity.real ../examples/URLClassification/url_train_2classes_weighted-rarity.pred -m 0
echo ""
echo "#######################################"

echo ""
echo "URL Classification training with rarity weights EXAMPLE - INVERSE FREQUENCY WEIGHTS IN WBA"
echo ""
python3 ../src/wba-evaluator.py ../examples/URLClassification/url_train_2classes_weighted-rarity.real ../examples/URLClassification/url_train_2classes_weighted-rarity.pred -m 2
echo ""
echo "#######################################"

echo ""
echo "URL Classification training with user weights EXAMPLE - BALANCED ACCURACY IN WBA"
echo ""
python3 ../src/wba-evaluator.py ../examples/URLClassification/url_train_2classes_weighted-user.real ../examples/URLClassification/url_train_2classes_weighted-user.pred -m 0
echo ""
echo "#######################################"

echo ""
echo "URL Classification training with user weights EXAMPLE - USER DEFINED WEIGHTS IN WBA"
echo ""
python3 ../src/wba-evaluator.py ../examples/URLClassification/url_train_2classes_weighted-user.real ../examples/URLClassification/url_train_2classes_weighted-user.pred -m 1 -w ../examples/URLClassification/url_2classes_user.weights
echo ""
echo "#######################################"
