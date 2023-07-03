echo ""
echo "#######################################"

echo ""
echo "LOG PARSING DRAIN macos EXAMPLE - BALANCED ACCURACY IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/macos.classdist ../examples/LogParsing/drainmacos.misclassify -c -m 0
echo ""
echo "#######################################"

echo ""
echo "LOG PARSING DRAIN macos EXAMPLE - INVERSE FREQUENCY WEIGHTS IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/macos.classdist ../examples/LogParsing/drainmacos.misclassify -c -m 2
echo ""
echo "#######################################"

echo ""
echo "***************************************"

echo ""
echo "LOG PARSING SPELL macos EXAMPLE - BALANCED ACCURACY IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/macos.classdist ../examples/LogParsing/spellmacos.misclassify -c -m 0
echo ""
echo "#######################################"

echo ""
echo "LOG PARSING SPELL macos EXAMPLE - INVERSE FREQUENCY WEIGHTS IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/macos.classdist ../examples/LogParsing/spellmacos.misclassify -c -m 2
echo ""
echo "#######################################"


echo ""
echo "***************************************"

echo ""
echo "LOG PARSING MOLFI macos EXAMPLE - BALANCED ACCURACY IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/macos.classdist ../examples/LogParsing/molfimacos.misclassify -c -m 0
echo ""
echo "#######################################"

echo ""
echo "LOG PARSING MOLFI macos EXAMPLE - INVERSE FREQUENCY WEIGHTS IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/macos.classdist ../examples/LogParsing/molfimacos.misclassify -c -m 2
echo ""
echo "#######################################"

