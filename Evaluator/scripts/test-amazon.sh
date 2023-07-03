echo ""
echo "#######################################"

echo ""
echo "AMAZON REVIEWS LSTM EXAMPLE - BALANCED ACCURACY IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/Amazon/amazonlstm.real ../examples/Amazon/amazonlstm.pred -m 0
echo ""
echo "#######################################"

echo ""
echo "AMAZON REVIEWS LSTM EXAMPLE - INVERSE FREQUENCY WEIGHTS IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/Amazon/amazonlstm.real ../examples/Amazon/amazonlstm.pred -m 2
echo ""
echo "#######################################"

echo ""
echo "AMAZON REVIEWS LSTM EXAMPLE - USER DEFINED WEIGHTS IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/Amazon/amazonlstm.real ../examples/Amazon/amazonlstm.pred -m 1 -w ../examples/Amazon/amazonuser.weights
echo ""
echo "#######################################"

echo ""
echo "AMAZON REVIEWS LSTM TRAIN EXAMPLE - BALANCED ACCURACY IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/Amazon/lstm_train.real ../examples/Amazon/lstm_train.pred -m 0
echo ""
echo "#######################################"

echo ""
echo "AMAZON REVIEWS LSTM TRAIN EXAMPLE - INVERSE FREQUENCY WEIGHTS IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/Amazon/lstm_train.real ../examples/Amazon/lstm_train.pred -m 2
echo ""
echo "#######################################"

echo ""
echo "AMAZON REVIEWS LSTM TRAIN EXAMPLE - USER DEFINED WEIGHTS IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/Amazon/lstm_train.real ../examples/Amazon/lstm_train.pred -m 1 -w ../examples/Amazon/amazonuser.weights
echo ""
echo "#######################################"

echo ""
echo "AMAZON REVIEWS LSTM TRAIN EXAMPLE - COMPOSITE(USER GIVEN + RARITY) WEIGHTS IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/Amazon/lstm_train.real ../examples/Amazon/lstm_train.pred -m 3 -w ../examples/Amazon/lstm_user.weights
echo ""
echo "#######################################"