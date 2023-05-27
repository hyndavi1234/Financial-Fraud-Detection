# Financial-Fraud-Detection
Implemented a fraud detection model utilizing Relational Graph Convolutional Networks to predict the fraudulent transactions using supervised learning techniques.

In recent years, financial fraud has become a significant problem for businesses and organizations worldwide. Financial fraud refers to illegal activities that involve intentionally deceiving people or institutions for personal gain. Due to the complexity and sophistication of fraudulent activities, detecting and preventing such activities has become a major challenge for financial institutions.

In this project proposal, we plan to explore the application of Graph Convolutional Networks (GCN) for detecting financial fraud. GCN is a type of neural network that can process data in the form of graphs, making it an ideal tool for detecting fraudulent activities in financial networks. Our goal is to develop a model that can accurately detect fraudulent activities in financial transactions and networks, using GCN.

The proposed project will involve the collection and analysis of financial transaction data, as well as the implementation and training of a GCN model for fraud detection. The potential impact of this project is significant, as it can help to detect and prevent financial fraud, ultimately leading to increased financial security and stability.

## Problem Description
Financial fraud is a significant challenge for businesses and organisations worldwide. The increasing complexity of fraudulent activities and the sheer volume of financial transactions make it challenging to detect fraudulent activities. The problem is com-
pounded by the fact that traditional fraud detection methods often rely on manually constructed rules, which may not be effective in detecting sophisticated fraud schemes. 

The proposed project aims to develop an application model for financial fraud detection that uses Graph Convolutional Networks (GCN). GCN is a type of neural network that can process data in the form of graphs, which is well-suited for detecting fraudulent activities in financial networks. The model will be trained on a dataset of financial transactions to learn patterns and features of fraudulent transactions. Here, Hetero-
geneous graph is defined as 𝐺 = (𝑉 , 𝐸, 𝑅), where (𝑣𝑖𝑉 ) represent the transaction or entity types, ((𝑣𝑖, 𝑟, 𝑣 𝑗)𝐸) represent the financial relationship between two entities, and (𝑟𝑅) define the type of relation.

The data labelling logic is the defined reported chargeback on the card as fraud transaction (isFraud=1) and transactions posterior to it with either user account, email address or billing address directly linked to these attributes as fraud too. If none of the above is reported and found beyond 120 days, then it is defined as legit transaction (isFraud=0). However, in real world fraudulent activity might not be reported, for example the cardholder was unaware, or forgot to report in time and beyond the claim period, etc. In such cases, supposed fraud might be labelled as legit, but never be known. Thus, there are various scenarios in determining whether the transaction is fraud or legit. There are various enriched features which represent the count of transaction properties like addresses, email addresses associated with the account, and attributes of match check like whether the purchaser and recipient first/last name match. These kinds of enriched or custom attributes along with the timestamps columns would help in detecting more fraud patterns. Our analysis on the data has shown such patterns.

The project addressed challenges, including the processing of large amounts of financial data, the detection of complex fraud schemes, and the usage of an effective RGCN model for fraud detection. Additionally, we have analysed the model sensitiveness to class imbalances.

<img width="500" alt="graph_network" src="https://github.com/hyndavi1234/Financial-Fraud-Detection/assets/34919619/12931e95-ef48-42cd-92a3-4a0e750a9e96">

## Datasets
The dataset contains the transactions that can be extracted from Kaggle, provided by Vesta Corporation, a leading payment service company whose data consists of verified transactions. This dataset has around 4 % of fraudulent transactions.

We are given a training set split into two CSV files. The first contains 590,000 observations of 394 transactional features; the second, 144,000 observations of 41 meta-identity features.

The transactional features span a broad gamut. Individual columns include the transaction amount, the time of transaction from some unspecified reference date, the product code for that transaction, and the purchaser and seller email domains. Grouped features contain information corresponding to transaction - card details, direct counts of attributes such as count of addresses associated with that account, time deltas, and matching indicators such as purchaser and recipient first/last name, and meta-features engineered.

The identity features refer broadly to digital signatures and network details collected from the transactions such as DeviceType, DeviceInfo, and IP address.

<table>
    <thead>
        <tr>
            <th colspan=2>Data Statistics</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>#Nodes</td>
            <td>105815</td>
        </tr>
        <tr>
            <td>#Edges</td>
            <td>3586258</td>
        </tr>
        <tr>
            <td>#Features Shape</td>
            <td>torch.Size([50000, 374])</td>
        </tr>
        <tr>
            <td>#Labeled Test Samples</td>
            <td>10000</td>
        </tr>
    </tbody>
</table>

## Methodology
The plan is to implement the application using Relational Graph Convolution Networks to predict the fraudulent transactions by generating unique features from the neighbourhood information to perform better predictions. After acquiring the data, prepossessing and edgelists generation using user identification columns are required. Afterwards, a multi-heterogeneous network is generated utilising the data and edgelists to predict fraudulence using Deep Learning models

### Relational Graph Convolutional Networks
RGCN - Relational Graph Convolution Networks is an extension of standard Graph Convolutional Networks designed to manage various relationships between entities in a knowledge base. It is mainly used to address entity categorization and link prediction
tasks respectively.

The key concept of RGCN is that its edges represent different relations. In contrast to GCN, each edge type in R-GCN is associated with a distinct matrix in order to learn unique representations for each relationship type. This represents relationships between
entities that are more sophisticated and intricate. RCGN has worked successfully in a wide range of applications including recommendation systems, social networks, and financial networks.

### RGCN in Financial Fraud Detection
This paradigm is largely motivated as an extension of GCNs that work on local graph neighbourhoods to large-scale relational data. These are designed specifically to deal with the highly multi-relational data present in realistic knowledge bases.

#### Graph Construction
The fundamental component of this fraud detection system is the creation of a graph from transactional data. The initial step is to determine the categorical and identity
attributes as well as the feature and non-feature columns. Following that, appropriate columns are selected that define the edges. These entity types are a subset of attributes in the transaction table and all of the attributes in the identity table that describe the identifying attribute in a transaction. For instance, kind of card (debit or credit), the IP address, the device ID, and the email domain.

During data preparation, the data for each identification attribute is recorded in csv files, with each file representing one kind of edge. These are also known as edge lists. Along with these, three other files were generated such as features.csv, tags,csv, and test.csv. The features.csv file contains the final transformed characteristics of the transaction nodes, tags.csv will include the class labels that are used for supervised training. Finally, test.csv will include the TransactionID data necessary to evaluate the performance of the model during testing.

The resultant edge lists are then used to construct the complete interaction graph using Deep Graph Library (DGL). In this scenario, each edge list represents a bipartite graph of transaction nodes and other entity types. The heterogeneous graph is constructed using a set of edge lists for each relation type and a feature matrix for the
nodes.

![img1](https://github.com/hyndavi1234/Financial-Fraud-Detection/assets/34919619/c4f577b7-12f8-49b3-98ed-a43e7ab22b31)

#### Hetero RGCN Model
Once the graph is constructed, its node features are normalised before being fed into the hetero-rgcn model. The goal is to learn node embeddings that capture the rela-
tionships between nodes in a graph. The model is initialised with the number of layers, hidden units, classes, edge and node types as input parameters.

The architecture of the model can be broken down into several components:
- Input layer: The input layer of an RGCN takes the feature vectors associated with each node in the graph as input. These feature vectors might represent information about the
node’s attributes (e.g. TransactionAmt, count of attribute values (such as count of addresses associated with the card)), or they might be learned embeddings based on text or other data associated with the node.
- Graph convolutional layers: The graph convolutional layers are the heart of the RGCN architecture. These layers perform convolutions over the graph, where the filters/weights are learned and applied based on the structure of the graph. Here,
each edge type is associated with a different weight matrix to learn different representations per relationship type.
- Activation Function: Each embedding of the convolution layer is passed through a non-linear activation function to adapt with new data and learn complex datasets to better fit the results with better accuracy.
- Output layer: The output layer of RGCN is a binary classifier using Linear Layer that produces the final output of the network. This is the prediction of a node’s label representing whether it is fraudulent or legitimate. In some cases, it might
be a learned representation of the graph that can be used for other downstream tasks.

![img2](https://github.com/hyndavi1234/Financial-Fraud-Detection/assets/34919619/ce824ed3-d7bd-4570-8def-a5283dfac248)

The graph convolutional layer in an RGCN contains the message passing scheme as follows:

```math
ℎ_𝑖 ^\left(𝑙+1\right) = 𝜎\left( \sum_{𝑟∈𝑅} \sum_{𝑗∈𝑁_𝑖^𝑟} {1 \over 𝑐_{𝑖,𝑟}} 𝑊_𝑟^\left(𝑙\right) ℎ_𝑗^\left(𝑙\right) + 𝑊_0^\left(𝑙\right) ℎ_𝑖^\left(𝑙\right) \right)
```

Here, ℎ<sub>𝑖</sub><sup>(𝑙)</sup> is the embedding of node 𝑖 in the 𝑙-th layer of the network. The superscript (𝑙+1) indicates the embedding in the next layer, and 𝜎 is an activation function and we are using Leaky 𝑅𝑒𝐿𝑈 in the implementation. 𝑅 is the set of edge types in the graph
(e.g.,"𝐷𝑒𝑣𝑖𝑐𝑒𝐼𝑛𝑓𝑜<>target", "𝑅𝑒𝑚𝑎𝑖𝑙𝑑𝑜𝑚𝑎𝑖𝑛<>target"), and 𝑁<sub>𝑖</sub><sup>𝑟</sup> is the set of neighbours of node 𝑖 connected by edges of type 𝑟 . 𝑊<sub>𝑟</sub><sup>(𝑙)</sup>  is a weight matrix associated with edge type 𝑟 at layer 𝑙, and 𝑐<sub>𝑖,𝑟</sub> is a normalisation constant that can be either learned or predefined which ensures the influence of each neighbour is weighted by the number of neighbours and it is problem specific.

In a nutshell, for a node, per relation type, first the message is generated by multiplying the nodes features with their corresponding edge weights [W\*h]. Then the messages from neighbours of that type are aggregated (mean). Then we trigger the message passing of multiple types followed by applying a non-linear activation function. This technique is repeated for additional layers, enabling the network to record higher-order interactions between entities. Finally a linear layer is applied to get a binary predicted value that represents whether a transaction is fraudulent or legitimate.

#### RGCN Models used
**1. Shallow RGCN:**
- Shallow RGCN is a variant of the RGCN architecture that contains a single convolution layer. It is a simple model that has shown impressive performance on various graph-based tasks such as node classification, link prediction, and graph classification. It can capture the local information in the network.
- The shallow RGCN model is composed of the input layer, hidden layer, and output layer. The input layer is responsible for transforming the node features into a format that can be used by the hidden layer. This is done by applying a set of learnable weight matrices to the node features. The hidden layer is responsible for aggregating the transformed node features and computing the final node embeddings. The resultant embeddings are finally parsed through a linear layer to output a binary predicted value indicating whether a transactionID is fraud or benign.
- One of the main advantages of this model is their simplicity and ease of use. The shallow RGCN model requires fewer parameters and is comparatively faster to train. Additionally, this can be effective for graph-based tasks involving small or moderately sized graphs.

**2. Deep RGCN:**
- Deep RGCN is another variant of the RGCN architecture that contains multiple hidden layers. The main idea behind this is to learn increasingly complex representations of the input node features by stacking multiple layers of message-passing and aggregation operations. It can capture both local and global information, allowing it to model complex interactions between distant nodes.
- In a deep RGCN, the information from the input features is propagated through multiple layers, with each layer refining the representations of the nodes. As a result, deep RGCNs are able to capture more nuanced relationships between the nodes in a graph and can potentially achieve better performance than shallow RGCNs.
- However, deep RGCNs are also more computationally expensive and as they propagate through multiple layers, it is difficult to learn the correct weights for the earlier layers due to gradients. 

Overall, the choice between shallow and deep RGCNs depends on the complexity of the graph, time computation, and the task at hand. For simpler graphs or tasks, a shallow RGCN may be sufficient, while for more complex graphs or tasks, a deep RGCN may be necessary to achieve high performance.

## Experiment Results
We have designed the network graph from the given transactional data and trained the HeteroRGCN model to detect the fraudulent transactions. The model is trained with a node embedding size of 360, 32 hidden size, for 200 epochs due to the computational challenges. The relevant hyper-parameters include a learning rate of 0.01, weight decay of 0.005, and a dropout rate of 0.2. We have tested the algorithms on Shallow RGCN, Deep RGCN, sampled datasets, other heterogeneous datasets and the experiment results are as follows:

**1. Shallow RGCN:**
- It has been trained on the complete data consisting of 500k transactions which constitute to around 720k nodes and 19M edges.
- The results obtained for Shallow RGCN trained with 3 layers has the following metrics: F1 score of 0.46, Accuracy of 0.97, Recall of 0.34

<img width="300" src="https://github.com/hyndavi1234/Financial-Fraud-Detection/assets/34919619/73fad460-8e4d-43b2-919f-ab4b58e4d6b8"/> <img width="300" src="https://github.com/hyndavi1234/Financial-Fraud-Detection/assets/34919619/aa34f68c-db44-4f2a-9fb2-41d77f48d084"/> <img width="300" src="https://github.com/hyndavi1234/Financial-Fraud-Detection/assets/34919619/723eae04-7d90-4c14-ac3b-9e4910e69a12"/>

**2. Deep RGCN:**
- It has also been trained on the complete data consisting of 500k transactions which constitute to around 720k nodes and 19M edges.
- The results obtained for this model trained with 6 layers has the following metrics:F1 score of 0.32, Accuracy of 0.97, Recall of 0.36
- If we execute this for more than 700 epochs then the model learns better and will increase its score of F1 up to 0.6

<img width="300" src="https://github.com/hyndavi1234/Financial-Fraud-Detection/assets/34919619/045528ac-c6ad-4449-b3ab-6664e3d1834e"/> <img width="300" src="https://github.com/hyndavi1234/Financial-Fraud-Detection/assets/34919619/4a739e14-0614-4c20-adf0-eea756bf6b8d"/> <img width="300" src="https://github.com/hyndavi1234/Financial-Fraud-Detection/assets/34919619/5d3463cf-1107-4cf7-a7b1-d189d1222f11"/>

**3. Class Imbalanced Test - Sampled Data:**
- Additionally, we have sampled the dataset to perform the class imbalance test which is the significant problem in the fraud detection systems. It has been trained on a subset of data consisting of 50k transactions which constitute to around 100k nodes and 3M edges.
- The fraud transactions considered in this data is 3% of sampled data resulting in up to 1500 transactions. This is essential in our case as it gives more insights about the class imbalances problem with less fraud transactions in the considered data.
- The results obtained for this model trained with 3 layers has the following metrics: F1 score of 0.88, Accuracy of 0.99, Recall of 0.79

<img width="300" src="https://github.com/hyndavi1234/Financial-Fraud-Detection/assets/34919619/ca164990-827b-45f5-b33d-363c9bccfba8"/> <img width="300" src="https://github.com/hyndavi1234/Financial-Fraud-Detection/assets/34919619/e2364544-51c3-429a-a9d8-5ddbc5b08235"/> <img width="300" src="https://github.com/hyndavi1234/Financial-Fraud-Detection/assets/34919619/36d6d71d-1088-4cea-b21f-de67941f040b"/>

**4. Yelp Review Fraud dataset**
- Moreover, we have modified our logic to incorporate other fraud datasets such as the Yelp review dataset available in DGL source which has multi-relational interactions that are having different relations between the same nodes.
- The implementation is parameterized allowing it to execute for different fraud datasets. It has a total of around 45k nodes and 8M edges. This test has been solely using the RGCN without any external logic for class imbalances.
- This dataset has been executed on 3 layers with 1000 epochs and following are the results obtained: F1 score of 0.35, Accuracy of 0.87, Recall of 0.30

<img width="300" src="https://github.com/hyndavi1234/Financial-Fraud-Detection/assets/34919619/25493c89-7565-4911-af99-57f005602e20"/> <img width="300" src="https://github.com/hyndavi1234/Financial-Fraud-Detection/assets/34919619/e07b033a-e160-45fd-8813-17f67eb77ca9"/> <img width="300" src="https://github.com/hyndavi1234/Financial-Fraud-Detection/assets/34919619/24872e70-5c18-48a0-8964-4085e8306782"/>

## Future Extensions
In the future, we plan to extend the proposed model to address some of the limitations and challenges faced in this fraud detection system.

In addition, we have analysed the ’Pick and Choose’ approach, which is designed to address class imbalance issues in financial networks. As per our initial analysis, the logic behind the pick option is related to the internal weight mechanism of RGCN. Our hypothesis is that addition of “choose” methodology to the existing system would increase its performance in detecting more fraudulent transactions. This would be our future extension to design a more robust model capable of handling class-imbalances to a greater extent.

## References
[1] Schlichtkrull, M., Kipf, T. N., Bloem, P., van den Berg, R., Titov, I., Welling, M. (2018). Modeling Relational Data with Graph Convolutional Networks. arXiv preprint arXiv:1703.06103.

[2] Liu, Y., Ao, X., Qin, Z., Chi, J., Feng, J., Yang, H., He, Q. (2021). Pick and Choose: A GNN-based Imbalanced Learning Approach for Fraud Detection. arXiv preprint arXiv:2108.06036.

[3] Dou, Y., Liu, Z., Sun, L., Deng, Y., Peng, H., Yu, P. S. (2020). Enhancing Graph Neural Network-based Fraud Detectors against Camouflaged Fraudsters. arXiv preprint arXiv:2009.03480.

[4] Adeshina, S. (2020). Detecting fraud in heterogeneous networks using Amazon SageMaker and Deep Graph Library. Retrieved from https://aws.amazon.com/blogs/machine-learning/detecting-fraud-in-heterogeneous-networks-using-amazon-sagemaker-and-deep-graph-library/.








