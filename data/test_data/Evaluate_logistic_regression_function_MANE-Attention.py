def computeLogRes_mane_attention():
   
    pathRead = "mane/" + dataset_choice + "/" + \
               method_choice[met] + "/"
    choice = 4
    n_splits = 5

    param = ""
    param_name = ""
    if dataset_choice == 'LinkedIn':

        sumf1macroList = []
        sumf1weightedList = []
        sumf1micro_maj_negList = []
        sumAUCList = []
        sumACCList = []

        OOsumf1macroList = []
        OOsumf1weightedList = []
        OOsumf1micro_maj_negList = []
        OOsumACCList = []
        OOsumAUCList = []
        split_cur = 1
        rep_avefmacro = 0
        rep_avefweighted = 0
        rep_avefmicro_maj_neg = 0
        rep_aveACC = 0
        rep_aveAUC = 0

        rep_OOavefmacro = 0
        rep_OOavefweighted = 0
        rep_OOavefmicro_maj_neg = 0
        rep_OOaveACC = 0
        rep_OOaveAUC = 0
        for rep in range(repeat):
            print(repeat)
            rep += 1
            sumf1macro = 0
            sumf1weighted = 0
            sumf1micro_maj_neg = 0
            sumACC = 0
            sumAUC = 0
            OOsumf1macro = 0
            OOsumf1weighted = 0
            OOsumf1micro_maj_neg = 0
            OOsumACC = 0
            OOsumAUC = 0
            for j in range(5):
                j += 1
                train_index = np.loadtxt(
                    pathTest + "/" + dataset_choice + "/train_indices" + str(rep) + str(
                        j) + ".txt",
                    dtype=int)  # integer format write
                test_index = np.loadtxt(
                    pathTest + "/" + dataset_choice + "/test_indices" + str(rep) + str(
                        j) + ".txt", dtype=int)

                node_embed_dict = {}
                File = pathRead + "/Embedding_concatenated" + str(j) + "_epoch_10_.txt"
                with open(File) as f:


                    for line in f:
                        key_protein = ((line.split()[0]))  ##str
                        embed = np.array(line.split()[1:])
                        node_embed_dict[key_protein] = np.array(embed).astype(
                            float)  # 
                two_nodes = np.column_stack((labels[:, 0], labels[:, 1]))
                label_data = list(map(int, labels[:, 2]))
                two_nodes_tuples = tuple(map(tuple, two_nodes))  # convert pairs to tuple
                tuple_label_dict = dict(zip(two_nodes_tuples, label_data))

                ############preparing embedding data based on given labelled pairs#############
                my_dictionary = OrderedDict()
                concat_emb = OrderedDict()
                label_emb = OrderedDict()
                for each_pair in two_nodes_tuples:
                    first_node = each_pair[0]
                    second_node = each_pair[1]
                    if first_node in node_embed_dict and second_node in node_embed_dict:  
                        first_node_emb = node_embed_dict[first_node]
                        second_node_emb = node_embed_dict[second_node]
                        concat_emb[each_pair] = np.concatenate((first_node_emb, second_node_emb))
                        label_emb[each_pair] = tuple_label_dict[each_pair]

                X = np.array(list(concat_emb.values()))

                
                y = np.array(list(map(int, label_emb.values())))
                n_classes = len(np.unique(y))

                c = Counter(y)
                class_list = np.unique(y)
                majority_class, value = c.most_common()[0]
                labels_to_consider = list(class_list)
                labels_to_consider.remove(majority_class)
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]

                classifier = OneVsRestClassifier(LogisticRegression())
                classifier.fit(X_train, y_train)
                predictions_labels = classifier.predict(X_test)
                predictions = classifier.predict_proba(X_test)[:, 1]

                # print("..............OneVsRest..............")
                sumf1macro += metrics.f1_score(y_test, predictions_labels, average='macro')
                sumf1macroList.append(metrics.f1_score(y_test, predictions_labels, average='macro'))
                sumf1weighted += metrics.f1_score(y_test, predictions_labels, average='weighted')
                sumf1weightedList.append(metrics.f1_score(y_test, predictions_labels, average='weighted'))
                sumf1micro_maj_neg += metrics.f1_score(y_test, predictions_labels, average='micro',
                                                       labels=labels_to_consider)
                sumf1micro_maj_negList.append(
                    metrics.f1_score(y_test, predictions_labels, average='micro', labels=labels_to_consider))
                sumACC += classifier.score(X_test, y_test)
                sumACCList.append(classifier.score(X_test, y_test))
                fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
                sumAUC += metrics.auc(fpr, tpr)
                sumAUCList.append(metrics.auc(fpr, tpr))
                # print("..............OneVsOne..............")
                classifier = OneVsOneClassifier(LogisticRegression())
                classifier.fit(X_train, y_train)
                predictions_labels = classifier.predict(X_test)

                results = OneVsOneClassifier(LogisticRegression()).fit(X_train, y_train)
                predictions_labels = results.predict(X_test)
                #   predictions = results.predict_proba(X_test)[:,1]

                OOsumf1macro += metrics.f1_score(y_test, predictions_labels, average='macro')
                OOsumf1macroList.append(metrics.f1_score(y_test, predictions_labels, average='macro'))
                OOsumf1weighted += metrics.f1_score(y_test, predictions_labels, average='weighted')
                OOsumf1weightedList.append(metrics.f1_score(y_test, predictions_labels, average='weighted'))
                OOsumf1micro_maj_neg += metrics.f1_score(y_test, predictions_labels, average='micro',
                                                         labels=labels_to_consider)
                print(metrics.f1_score(y_test, predictions_labels, average='micro', labels=labels_to_consider))
                OOsumf1micro_maj_negList.append(
                    metrics.f1_score(y_test, predictions_labels, average='micro', labels=labels_to_consider))
                OOsumACC += classifier.score(X_test, y_test)
                OOsumACCList.append(classifier.score(X_test, y_test))
                fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions_labels, pos_label=1)
                OOsumAUC += metrics.auc(fpr, tpr)
                OOsumAUCList.append(metrics.auc(fpr, tpr))
                split_cur += 1
            avefmacro = sumf1macro / n_splits
            avefweighted = sumf1weighted / n_splits
            avefmicro_maj_neg = sumf1micro_maj_neg / n_splits
            aveACC = sumACC / n_splits
            aveAUC = sumAUC / n_splits

            rep_avefmacro += avefmacro
            rep_avefweighted += avefweighted
            rep_avefmicro_maj_neg += avefmicro_maj_neg
            rep_aveACC += aveACC
            rep_aveAUC += aveAUC

            OOavefmacro = OOsumf1macro / n_splits
            OOavefweighted = OOsumf1weighted / n_splits
            OOavefmicro_maj_neg = OOsumf1micro_maj_neg / n_splits
            OOaveACC = OOsumACC / n_splits
            OOaveAUC = OOsumAUC / n_splits

            rep_OOavefmacro += OOavefmacro
            rep_OOavefweighted += OOavefweighted
            rep_OOavefmicro_maj_neg += OOavefmicro_maj_neg
            rep_OOaveACC += OOaveACC
            rep_OOaveAUC += OOaveAUC
        n_splits = repeat
        avefmacro = rep_avefmacro / n_splits
        avefweighted = rep_avefweighted / n_splits
        avefmicro_maj_neg = rep_avefmicro_maj_neg / n_splits
        aveACC = rep_aveACC / n_splits
        aveAUC = rep_aveAUC / n_splits

        OOavefmacro = rep_OOavefmacro / n_splits
        OOavefweighted = rep_OOavefweighted / n_splits
        OOavefmicro_maj_neg = rep_OOavefmicro_maj_neg / n_splits
        OOaveACC = rep_OOaveACC / n_splits
        OOaveAUC = rep_OOaveAUC / n_splits

        sumf1macroList = np.array(sumf1macroList)
        sumf1weightedList = np.array(sumf1weightedList)
        sumf1micro_maj_negList = np.array(sumf1micro_maj_negList)
        sumAUCList = np.array(sumAUCList)
        sumACCList = np.array(sumACCList)

        OOsumf1macroList = np.array(OOsumf1macroList)
        OOsumf1weightedList = np.array(OOsumf1weightedList)
        OOsumf1micro_maj_negList = np.array(OOsumf1micro_maj_negList)
        OOsumAUCList = np.array(OOsumAUCList)
        OOsumACCList = np.array(OOsumACCList)

        new_file = open("AUC5X5test" + dataset_choice + ".txt", mode="a+")
        new_file.write(
            "Version " + "Method " + param_name + " fMicro " + "fMacro " + "fWeighted " + "accuracy " + "stdefMicro " + "stdefMacro " + "stdefWeighted " + "stdeaccuracy" + " stdfMicro " + "stdfMacro " + "stdfWeighted " + "stdaccuracy" + '\n')

        ALL_RESULT = 'One-vs-Rest: ' + method_choice[met] + " " + str(param) + ' ' + str(avefmicro_maj_neg) + ' ' + str(
            avefmacro) + ' ' + str(avefweighted) + ' ' + str(aveACC) + ' ' + str(
            np.std(sumf1micro_maj_negList, ddof=1) / np.sqrt(25)) + ' ' + str(
            np.std(sumf1macroList, ddof=1) / np.sqrt(25)) + ' ' + str(
            np.std(sumf1weightedList, ddof=1) / np.sqrt(25)) + ' ' + str(
            np.std(sumACCList, ddof=1) / np.sqrt(25)) + ' ' + str(np.std(sumf1micro_maj_negList, ddof=1)) + ' ' + str(
            np.std(sumf1macroList, ddof=1)) + ' ' + str(np.std(sumf1weightedList, ddof=1)) + ' ' + str(
            np.std(sumACCList, ddof=1)) + '\n'
        print(ALL_RESULT)
        new_file.write(ALL_RESULT)

        ALL_RESULT = 'One-vs-One: ' + method_choice[met] + " " + str(param) + ' ' + str(
            OOavefmicro_maj_neg) + ' ' + str(
            OOavefmacro) + ' ' + str(OOavefweighted) + ' ' + str(OOaveACC) + ' ' + str(
            np.std(OOsumf1micro_maj_negList, ddof=1) / np.sqrt(25)) + ' ' + str(
            np.std(OOsumf1macroList, ddof=1) / np.sqrt(25)) + ' ' + str(
            np.std(OOsumf1weightedList, ddof=1) / np.sqrt(25)) + ' ' + str(
            np.std(OOsumACCList, ddof=1) / np.sqrt(25)) + ' ' + str(
            np.std(OOsumf1micro_maj_negList, ddof=1)) + ' ' + str(
            np.std(OOsumf1macroList, ddof=1)) + ' ' + str(np.std(OOsumf1weightedList, ddof=1)) + ' ' + str(
            np.std(OOsumACCList, ddof=1)) + '\n'
        print(ALL_RESULT)
        new_file.write(ALL_RESULT)
        new_file.close()


    if dataset_choice == 'youtube':

        sumAUCList = []
        sumAUCPRList = []

        sumaverages = 0
        sumaveragesAUCPR = 0
        for rep in range(repeat):
            rep += 1
            sumAUC = 0
            sumAUCPR = 0
            split_cur = 1
            for j in range(5):
                j += 1
                train_index = np.loadtxt(
                    pathTest + "/" + dataset_choice + "/train_indices" + str(rep) + str(
                        j) + ".txt",
                    dtype=int)  
                test_index = np.loadtxt(
                    pathTest + "/" + dataset_choice + "/test_indices" + str(rep) + str(
                        j) + ".txt", dtype=int)
                node_embed_dict = {}
                File = pathRead + "Embedding_concatenated" + str(j) + "_epoch_10_.txt"

                with open(File) as f:

                    for line in f:
                        key_protein = ((line.split()[0]))  
                        embed = np.array(line.split()[1:])
                        node_embed_dict[key_protein] = np.array(embed).astype(float)
                two_nodes = np.column_stack((labels[:, 0], labels[:, 1]))
                label_data = list(map(int, labels[:, 2]))
                two_nodes_tuples = tuple(map(tuple, two_nodes))  
                tuple_label_dict = dict(zip(two_nodes_tuples, label_data))

                ############preparing embedding data based on given labelled pairs#############
                my_dictionary = OrderedDict()
                concat_emb = OrderedDict()
                label_emb = OrderedDict()
                for each_pair in two_nodes_tuples:
                    first_node = each_pair[0]
                    second_node = each_pair[1]
                    if first_node in node_embed_dict and second_node in node_embed_dict:  

                        first_node_emb = node_embed_dict[first_node]
                        second_node_emb = node_embed_dict[second_node]
                        concat_emb[each_pair] = np.concatenate((first_node_emb, second_node_emb))
                        label_emb[each_pair] = tuple_label_dict[each_pair]


                X = np.array(list(concat_emb.values()))

                y = np.array(list(map(int, label_emb.values())))
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                logisticRegr = LogisticRegression()
                logisticRegr.fit(X_train, y_train)
                predictions = logisticRegr.predict_proba(X_test)[:, 1]

                
                fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
                sumAUC += metrics.auc(fpr, tpr)
                sumAUCList.append(metrics.auc(fpr, tpr))
                precision, recall, thresholds = metrics.precision_recall_curve(y_test, predictions)

                sumAUCPR += metrics.average_precision_score(y_test, predictions)
                sumAUCPRList.append(metrics.average_precision_score(y_test, predictions))
                split_cur += 1

            average = sumAUC / n_splits
            averageAUPRC = sumAUCPR / n_splits

            sumaverages += average
            sumaveragesAUCPR += averageAUPRC
        sumAUCList = np.array(sumAUCList)
        sumAUCPRList = np.array(sumAUCPRList)
        n_splits = repeat
        average = sumaverages / n_splits
        averageAUPRC = sumaveragesAUCPR / n_splits

        new_file = open("ndcg" + str(repeat) + "X5" + dataset_choice + ".txt", mode="a+")
        new_file.write("Method " + param_name + " AUC " + "stdeAUC " + "stdeAUPRC" + " stdAUC " + " stdAUPRC" + '\n')

        ALL_RESULT = method_choice[met] + " " + str(param) + ' ' + str(average) + ' ' + str(averageAUPRC) + ' ' + str(
            np.std(sumAUCList, ddof=1) / np.sqrt(25)) + ' ' + str(
            np.std(sumAUCPRList, ddof=1) / np.sqrt(25)) + ' ' + str(np.std(sumAUCList, ddof=1)) + ' ' + str(
            np.std(sumAUCPRList, ddof=1)) + '\n'
        print(ALL_RESULT)
        new_file.write(ALL_RESULT)
        new_file.close()


    if dataset_choice == 'IntAct':

        sum_disease_AUC = 0
        sum_disease_AUCList = []
        sum_disease_AUPRC = 0
        sum_disease_AUPRCList = []
        sum_cancerdisease_AUPRC = 0
        sum_cancerdisease_AUC = 0
        out_folder = dataset_choice + 'scores' + str(repeat) + 'x5folds' + method_choice[
            met] + "" + param_name + "" + str(param)

        new_file = open("ndcg" + str(repeat) + "X5" + dataset_choice + ".txt", mode="a+")
        new_file.write(
            "Method " + param_name + " Disease " + "p@" + str(top_k) + " " + "r@" + str(top_k) + " " + "ap@" + str(
                top_k) + " " + "map " + "ndcg@" + str(
                top_k) + " " + "balanced_acc. " + "AUC " + "scikit_AreaPR " + "AUPRC " + "stdeAUC " + "stdeAUPRC" + " stdAUC " + " stdAUPRC" + '\n')

        sum_disease_areaPR = 0
        sum_disease_b_acc = 0

        sum_disease_ndcg = 0
        sum_disease_pk = 0
        sum_disease_rk = 0
        sum_disease_apk = 0
        sum_disease_mapk = 0
        for dis in range(len(diseases)):  
            disease = diseases[dis]
            sumAUCList = []
            sumAUCPRList = []
            sumaverages = 0
            sumaveragesAUCPR = 0
            sumaverages_ndcg = 0
            sumaverages_balanced_accuracy = 0
            for rep in range(repeat):
                rep += 1
                sumaverages_ndcg = 0
                sumaverages_pk = 0
                sumaverages_rk = 0
                sumaverages_apk = 0
                sumaverages_mapk = 0
                sumaverages_b_acc = 0

                sumaverages_areaPR = 0
                sum_ndcg = 0
                sum_pk = 0
                sum_rk = 0
                sum_apk = 0
                sum_mapk = 0

                sumAUC = 0
                sumAUCPR = 0
                sum_areaPR = 0
                sum_b_acc = 0
                for j in range(5):
                    j += 1
                    node_embed_dict = {}
                    # idx2node = {idx: str(node) for (idx, node) in enumerate(common_node_names)}
                    File = pathRead + "Embedding_concatenated" + str(j) + "_epoch_10_.txt"
                    with open(File) as f:
                        
                        for line in f:
                            key_protein = int(line.split()[0])
                            embed = np.array(line.split()[1:])
                            node_embed_dict[str(key_protein)] = np.array(embed).astype(
                                float)  

                    trainIndices = []
                    train_IndiceFile = pathTest + "/" + dataset_choice + "/" + disease + "" + dataset + "" + str(
                        rep) + "" + str(j) + "indicesofTrainSet.txt"
                    fR = open(train_IndiceFile, 'r')
                    for line in fR:
                        trainIndices.append(int(line.strip()))
                    xtrain = []  

                    train_label_file = pathTest + "/" + dataset_choice + "/" + disease + dataset + "" + str(
                        rep) + "" + str(j) + "trainLabel.txt"
                    ytrain = []
                    fR = open(train_label_file, 'r')

                    index = 0
                    trainindicesused = []
                    for line in fR:  # as reading train labels
                        i = trainIndices[index]  # i is node name in ppi, since in ppi nodes reprsented by indices
                        i = str(i)
                        if i in node_embed_dict:  # for nodes which do not have any neighbors in entire dataset to be trained,i.e, to check singleton nodes in all views
                            xtrain.append(node_embed_dict[i])
                            ytrain.append(int(line.strip()[0]))  # if it is in xtrain then it must also be in ytrain
                            trainindicesused.append(int(i))
                        index += 1

                    xtrain = np.array(xtrain)
                    print(xtrain.shape)
                    ytrain = np.array(ytrain)
                    trainindicesused = np.array(trainindicesused)

                    testIndices = []
                    test_IndiceFile = pathTest + "/" + dataset_choice + "/" + disease + "" + dataset + "" + str(
                        rep) + "" + str(j) + "indicesofTestSet.txt"
                    fR = open(test_IndiceFile, 'r')
                    for line in fR:
                        testIndices.append(int(line.strip()))
                    xtest = []  

                    test_label_file = pathTest + "/" + dataset_choice + "/" + disease + dataset + "" + str(
                        rep) + "" + str(j) + "testLabel.txt"
                    ytest = []
                    fR = open(test_label_file, 'r')

                    
                    index = 0
                    testindicesused = []

                    for line in fR:
                        i = testIndices[index]
                        i = str(i)
                        if i in node_embed_dict:  # for nodes which do not have neighbors to be trained
                            xtest.append(node_embed_dict[i])
                            
                            ytest.append(int(line.strip()[0]))
                            testindicesused.append(int(i))
                        index += 1
                    xtest = np.array(xtest)
                    ytest = np.int_(np.array(ytest))
                    testindicesused = np.array(testindicesused)
                    log_reg = LogisticRegression()
                    log_reg.fit(xtrain, ytrain)
                    predictions = log_reg.predict_proba(xtest)[:, 1]

                    AUC, AUCPR, areaPR, b_acc = survey_methods_eval(ytest, predictions)
                    sumAUC += AUC
                    sumAUCPR += AUCPR

                    indices = np.argsort(-predictions)  # minus decreasing
                    predictions_init = np.sort(predictions)[::-1]  # [::-1] converts to reverse order (decreasing)
                    ytest_init = ytest[indices]
                    print(ytest_init)
                    actual = [i for i in range(len(ytest_init)) if ytest_init[i] == 1]
                    print(actual)
                    sum_ndcg += ndcg_at_k(ytest_init, top_k)
                    sum_pk += pk(actual, top_k)
                    sum_rk += rk(actual, top_k)
                    sum_apk += apk(actual, top_k)
                    sum_mapk += mapk(actual, len(ytest))
                    sum_b_acc += b_acc
                    sum_areaPR += areaPR

                average = sumAUC / n_splits
                averageAUPRC = sumAUCPR / n_splits
                sumaverages += average
                sumaveragesAUCPR += averageAUPRC

                average_ndcg = sum_ndcg / n_splits
                sumaverages_ndcg += average_ndcg
                average_pk = sum_pk / n_splits
                sumaverages_pk += average_pk
                average_rk = sum_rk / n_splits
                sumaverages_rk += average_rk
                average_apk = sum_apk / n_splits
                sumaverages_apk += average_apk
                average_mapk = sum_mapk / n_splits
                sumaverages_mapk += average_mapk

                average_areaPR = sum_areaPR / n_splits
                sumaverages_areaPR += average_areaPR
                average_b_acc = sum_b_acc / n_splits
                sumaverages_b_acc += average_b_acc

            AUCresult = sumaverages / repeat
            AUCPRresult = sumaveragesAUCPR / repeat
            sumaverages_ndcg_result = sumaverages_ndcg / repeat
            sumaverages_pk_result = sumaverages_pk / repeat
            sumaverages_rk_result = sumaverages_rk / repeat
            sumaverages_apk_result = sumaverages_apk / repeat
            sumaverages_mapk_result = sumaverages_mapk / repeat

            sumaverages_areaPR_result = sumaverages_areaPR / repeat
            sumaverages_b_acc_result = sumaverages_b_acc / repeat

            ALL_RESULT = method_choice[met] + " " + str(param) + " " + disease + ' ' + str(
                sumaverages_pk_result) + ' ' + str(sumaverages_rk_result) + ' ' + str(
                sumaverages_apk_result) + ' ' + str(sumaverages_mapk_result) + ' ' + str(
                sumaverages_ndcg_result) + ' ' + str(sumaverages_b_acc_result) + ' ' + str(AUCresult) + ' ' + str(
                sumaverages_areaPR_result) + ' ' + str(AUCPRresult) + ' ' + str(
                np.std(sumAUCList, ddof=1) / np.sqrt(prev_splits * repeat)) + ' ' + str(
                np.std(sumAUCPRList, ddof=1) / np.sqrt(prev_splits * repeat)) + ' ' + str(
                np.std(sumAUCList, ddof=1)) + ' ' + str(np.std(sumAUCPRList, ddof=1)) + '\n'
            print(ALL_RESULT)
            new_file.write(ALL_RESULT)

            sum_disease_AUPRC += AUCPRresult
            sum_disease_AUC += AUCresult
            sum_disease_ndcg += sumaverages_ndcg_result

            sum_disease_pk += sumaverages_pk_result
            sum_disease_rk += sumaverages_rk_result
            sum_disease_apk += sumaverages_apk_result
            sum_disease_mapk += sumaverages_mapk_result

            sum_disease_areaPR += sumaverages_areaPR_result
            sum_disease_b_acc += sumaverages_b_acc_result


        ALL_RESULT = method_choice[met] + " " + str(param) + " " + 'average_diseases' + ' ' + str(
            sum_disease_pk / len(diseases)) + ' ' + str(sum_disease_rk / len(diseases)) + ' ' + str(
            sum_disease_apk / len(diseases)) + ' ' + str(sum_disease_mapk / len(diseases)) + ' ' + str(
            sum_disease_ndcg / len(diseases)) + ' ' + str(sum_disease_b_acc / len(diseases)) + ' ' + str(
            sum_disease_AUC / len(diseases)) + ' ' + str(sum_disease_areaPR / len(diseases)) + ' ' + str(
            sum_disease_AUPRC / len(diseases)) + ' ' + str(
            np.std(sum_disease_AUCList, ddof=1) / np.sqrt(175)) + ' ' + str(
            np.std(sum_disease_AUPRCList, ddof=1) / np.sqrt(175)) + ' ' + str(
            np.std(sum_disease_AUCList, ddof=1)) + ' ' + str(np.std(sum_disease_AUPRCList, ddof=1)) + '\n'
        print(ALL_RESULT)
        new_file.write(ALL_RESULT)

        new_file.close()
