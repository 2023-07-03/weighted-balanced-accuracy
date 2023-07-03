echo ""
echo "#######################################"

echo ""
echo "LOG PARSING DRAIN HDFS EXAMPLE - BALANCED ACCURACY IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/hdfs.classdist ../examples/LogParsing/drainhdfs.misclassify -c -m 0
echo ""
echo "#######################################"

echo ""
echo "LOG PARSING DRAIN HDFS EXAMPLE - INVERSE FREQUENCY WEIGHTS IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/hdfs.classdist ../examples/LogParsing/drainhdfs.misclassify -c -m 2
echo ""
echo "#######################################"


echo ""
echo "***************************************"

echo ""
echo "LOG PARSING SPELL HDFS EXAMPLE - BALANCED ACCURACY IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/hdfs.classdist ../examples/LogParsing/spellnhdfs.misclassify -c -m 0
echo ""
echo "#######################################"

echo ""
echo "LOG PARSING SPELL HDFS EXAMPLE - INVERSE FREQUENCY WEIGHTS IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/hdfs.classdist ../examples/LogParsing/spellhdfs.misclassify -c -m 2
echo ""
echo "#######################################"



echo ""
echo "***************************************"

echo ""
echo "LOG PARSING MOLFI HDFS EXAMPLE - BALANCED ACCURACY IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/hdfs.classdist ../examples/LogParsing/molfihdfs.misclassify -c -m 0
echo ""
echo "#######################################"

echo ""
echo "LOG PARSING MOLFI HDFS EXAMPLE - INVERSE FREQUENCY WEIGHTS IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/hdfs.classdist ../examples/LogParsing/molfihdfs.misclassify -c -m 2
echo ""
echo "#######################################"


