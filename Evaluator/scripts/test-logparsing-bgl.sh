echo ""
echo "#######################################"

echo ""
echo "LOG PARSING DRAIN bgl EXAMPLE - BALANCED ACCURACY IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/bgl.classdist ../examples/LogParsing/drainbgl.misclassify -c -m 0
echo ""
echo "#######################################"

echo ""
echo "LOG PARSING DRAIN bgl EXAMPLE - INVERSE FREQUENCY WEIGHTS IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/bgl.classdist ../examples/LogParsing/drainbgl.misclassify -c -m 2
echo ""
echo "#######################################"


echo ""
echo "***************************************"

echo ""
echo "LOG PARSING SPELL bgl EXAMPLE - BALANCED ACCURACY IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/bgl.classdist ../examples/LogParsing/spellbgl.misclassify -c -m 0
echo ""
echo "#######################################"

echo ""
echo "LOG PARSING SPELL bgl EXAMPLE - INVERSE FREQUENCY WEIGHTS IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/bgl.classdist ../examples/LogParsing/spellbgl.misclassify -c -m 2
echo ""
echo "#######################################"


echo ""
echo "***************************************"

echo ""
echo "LOG PARSING MOLFI bgl EXAMPLE - BALANCED ACCURACY IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/bgl.classdist ../examples/LogParsing/molfibgl.misclassify -c -m 0
echo ""
echo "#######################################"

echo ""
echo "LOG PARSING MOLFI bgl EXAMPLE - INVERSE FREQUENCY WEIGHTS IN WBA"
echo ""
python ../src/wba-evaluator.py ../examples/LogParsing/bgl.classdist ../examples/LogParsing/molfibgl.misclassify -c -m 2
echo ""
echo "#######################################"
