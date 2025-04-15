
##############################################################################
PathQuestion datasets
----------------------
https://github.com/zmtkeke/IRN
-Here, you can also find the Knowledge Graph for each sub-datasets
###########################################################################################

##############################################################################
Multiple-choice questions
----------------------
https://github.com/EchoDreamer/Q-KGR
-We used the preprocessed versions created by the authors of DRAGON and available at:https://nlp.stanford.edu/projects/myasu/DRAGON/data_preprocessed.zip
-These preprocessed versions all use ConceptNet as underlying Knowledge Graph, which is also provided.
###########################################################################################



##############################################################################
Comparison with iterative KG-LLM systems
----------------------
ZeroshorRE and ComplexWebQuestions are available here: https://github.com/GasolSun36/ToG/tree/main/data
###########################################################################################


###
Splits: For each datasets the splits are provided.
###


All datasets need to be parsed to obtain the input for GraLan. To do so, first call GraphBuilder; it will parse and create.pkl files for the Kg, dictionaries and the  predicate relatedness matrix. Then, call QuestionDataset, which will construct the final datasets.

