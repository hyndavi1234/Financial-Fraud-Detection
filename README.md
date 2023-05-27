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

<img width="131" alt="graph_network" src="https://github.com/hyndavi1234/Financial-Fraud-Detection/assets/34919619/12931e95-ef48-42cd-92a3-4a0e750a9e96">

## Datasets
The dataset contains the transactions that can be extracted from Kaggle, provided by Vesta Corporation, a leading payment service company whose data consists of verified transactions. This dataset has around 4 % of fraudulent transactions.

We are given a training set split into two CSV files. The first contains 590,000 observations of 394 transactional features; the second, 144,000 observations of 41 meta-identity features.

The transactional features span a broad gamut. Individual columns include the transaction amount, the time of transaction from some unspecified reference date, the product code for that transaction, and the purchaser and seller email domains. Grouped features contain information corresponding to transaction - card details, direct counts of attributes such as count of addresses associated with that account, time deltas, and matching indicators such as purchaser and recipient first/last name, and meta-features engineered.

The identity features refer broadly to digital signatures and network details collected from the transactions such as DeviceType, DeviceInfo, and IP address.

## ADD TABLE HERE

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

## ADD THE IMAGE HERE

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

The graph convolutional layer in an RGCN contains the message passing scheme as follows:
ℎ(𝑙+1)
𝑖 = 𝜎 (Í𝑟 ∈𝑅
Í𝑗 ∈𝑁 𝑟
𝑖
1
𝑐𝑖,𝑟 𝑊 (𝑙 )
𝑟 ℎ(𝑙 )
𝑗 ) + 𝑊 (𝑙 )
0 ℎ(𝑙 )
𝑖 )
Here, ℎ(𝑙 )𝑖 is the embedding of node 𝑖 in the 𝑙-th layer of the network. The superscript (𝑙 + 1) indicates the embedding in the next layer, and 𝜎 is an activation function and we are using Leaky 𝑅𝑒𝐿𝑈 in the implementation. 𝑅 is the set of edge types in the graph
(e.g.,"𝐷𝑒𝑣𝑖𝑐𝑒𝐼𝑛𝑓𝑜<>target", "𝑅𝑒𝑚𝑎𝑖𝑙𝑑𝑜𝑚𝑎𝑖𝑛<>target"), and 𝑁 𝑟𝑖 is the set of neighbours of node 𝑖 connected by edges of type 𝑟 . 𝑊 (𝑙 ) 𝑟 is a weight matrix associated with edge type 𝑟 at layer 𝑙, and 𝑐𝑖,𝑟 is a normalisation constant that can be either learned or predefined which ensures the influence of each neighbour is weighted by the number of neighbours and it is problem specific.

In a nutshell, for a node, per relation type, first the message is generated by multiplying the nodes features with their corresponding edge weights [W\*h]. Then the messages from neighbours of that type are aggregated (mean). Then we trigger the message passing of multiple types followed by applying a non-linear activation function. This technique is repeated for additional layers, enabling the network to record higher-order interactions between entities. Finally a linear layer is applied to get a binary predicted value that represents whether a transaction is fraudulent or legitimate.
