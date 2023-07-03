echo ""
echo "#######################################"

echo ""
echo "LOG PARSING DRAIN android EXAMPLE - BALANCED ACCURACY IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/android.classdist ../examples/LogParsing/drainandroid.misclassify -c -m 0
echo ""
echo "#######################################"

echo ""
echo "LOG PARSING DRAIN android EXAMPLE - INVERSE FREQUENCY WEIGHTS IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/android.classdist ../examples/LogParsing/drainandroid.misclassify -c -m 2
echo ""
echo "#######################################"

echo ""
echo "***************************************"

echo ""
echo "LOG PARSING SPELL android EXAMPLE - BALANCED ACCURACY IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/android.classdist ../examples/LogParsing/spellandroid.misclassify -c -m 0
echo ""
echo "#######################################"

echo ""
echo "LOG PARSING SPELL android EXAMPLE - INVERSE FREQUENCY WEIGHTS IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/android.classdist ../examples/LogParsing/spellandroid.misclassify -c -m 2
echo ""
echo "#######################################"


echo ""
echo "***************************************"

echo ""
echo "LOG PARSING MOLFI android EXAMPLE - BALANCED ACCURACY IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/android.classdist ../examples/LogParsing/molfiandroid.misclassify -c -m 0
echo ""
echo "#######################################"

echo ""
echo "LOG PARSING MOLFI android EXAMPLE - INVERSE FREQUENCY WEIGHTS IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/android.classdist ../examples/LogParsing/molfiandroid.misclassify -c -m 2
echo ""
echo "#######################################"


