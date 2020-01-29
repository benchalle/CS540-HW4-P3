import java.util.*;

/**
 * The main class that handles the entire network
 * Has multiple attributes each with its own use
 */

public class NNImpl {
    private ArrayList<Node> inputNodes; //list of the output layer nodes.
    private ArrayList<Node> hiddenNodes;    //list of the hidden layer nodes
    private ArrayList<Node> outputNodes;    // list of the output layer nodes

    private ArrayList<Instance> trainingSet;    //the training set

    private double learningRate;    // variable to store the learning rate
    private int maxEpoch;   // variable to store the maximum number of epochs
    private Random random;  // random number generator to shuffle the training set

    /**
     * This constructor creates the nodes necessary for the neural network
     * Also connects the nodes of different layers
     * After calling the constructor the last node of both inputNodes and
     * hiddenNodes will be bias nodes.
     */

    NNImpl(ArrayList<Instance> trainingSet, int hiddenNodeCount, Double learningRate, int maxEpoch, Random random, Double[][] hiddenWeights, Double[][] outputWeights) {
        this.trainingSet = trainingSet;
        this.learningRate = learningRate;
        this.maxEpoch = maxEpoch;
        this.random = random;

        //input layer nodes
        inputNodes = new ArrayList<>();
        int inputNodeCount = trainingSet.get(0).attributes.size();
        int outputNodeCount = trainingSet.get(0).classValues.size();
        for (int i = 0; i < inputNodeCount; i++) {
            Node node = new Node(0);
            inputNodes.add(node);
        }

        //bias node from input layer to hidden
        Node biasToHidden = new Node(1);
        inputNodes.add(biasToHidden);

        //hidden layer nodes
        hiddenNodes = new ArrayList<>();
        for (int i = 0; i < hiddenNodeCount; i++) {
            Node node = new Node(2);
            //Connecting hidden layer nodes with input layer nodes
            for (int j = 0; j < inputNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(inputNodes.get(j), hiddenWeights[i][j]);
                node.parents.add(nwp);
            }
            hiddenNodes.add(node);
        }

        //bias node from hidden layer to output
        Node biasToOutput = new Node(3);
        hiddenNodes.add(biasToOutput);

        //Output node layer
        outputNodes = new ArrayList<>();
        for (int i = 0; i < outputNodeCount; i++) {
            Node node = new Node(4);
            //Connecting output layer nodes with hidden layer nodes
            for (int j = 0; j < hiddenNodes.size(); j++) {
                NodeWeightPair nwp = new NodeWeightPair(hiddenNodes.get(j), outputWeights[i][j]);
                node.parents.add(nwp);
            }
            outputNodes.add(node);
        }
    }

    /**
     * Get the prediction from the neural network for a single instance
     * Return the idx with highest output values. For example if the outputs
     * of the outputNodes are [0.1, 0.5, 0.2], it should return 1.
     * The parameter is a single instance
     */

    public int predict(Instance instance) {
    	int i = 0;
    	for(Double input: instance.attributes) {
    		inputNodes.get(i).setInput(input);
    		i++;
    	}
    	
    	for(Node node : hiddenNodes) {
			node.calculateOutput();
		}

		double sum= 0;
		for(Node node : outputNodes) {
			node.calculateOutput();
			sum += node.getOutput();
		}
		for(Node node : outputNodes) {
			node.calculateOutput(sum);
		}
		
    	double six = outputNodes.get(0).getOutput();
    	double eight = outputNodes.get(1).getOutput();
    	double nine = outputNodes.get(2).getOutput();
    	double sixOrEight = Math.max(six, eight);
    	if(sixOrEight == six) {
    	double sixOrNine = Math.max(six, nine);
    		if(sixOrNine == six) {
    			return 0;
    		}else {
    			return 2;
    		}
    	}else {
    		double eightOrNine = Math.max(eight, nine);
    		if(eightOrNine == eight) {
    			return 1;
    		}else {
    			return 2;
    		}
    	}
    }


    /**
     * Train the neural networks with the given parameters
     * <p>
     * The parameters are stored as attributes of this class
     */

    public void train() {
    	double totalLoss = 0;
    	for(int epoch = 0; epoch < maxEpoch; epoch++) {
    		Collections.shuffle(trainingSet,random);
    		//double lossBefore = 0;
    		for(Instance inst : trainingSet){
    			for (int j = 0; j < inputNodes.size()-1; j++) {
    				inputNodes.get(j).setInput(inst.attributes.get(j));
    			}

    			for(Node node : hiddenNodes) {
    				node.calculateOutput();
    			}
    					//output level
    			double sum= 0;
    			for(Node node : outputNodes) {
    				node.calculateOutput();
    				sum += node.getOutput();
    			}
    			for(Node node : outputNodes) {
   					node.calculateOutput(sum);
   				}

    			int index = 0;
    			for(Node node : outputNodes) {
    				node.calculateDelta(outputNodes, inst.classValues.get(index));
    				index++;
    			}
    			
    			for(Node node : hiddenNodes) {
    				node.calculateDelta(outputNodes, 0);
    			}
    		
    			//lossBefore = -loss(trainingSet.get(epoch));
    			//System.out.println("LossBefore: " + String.format("%.3e",lossBefore));
    		
    			for(Node node : outputNodes) {
    				node.updateWeight(learningRate);
    			} 
    			
        		for(Node node : hiddenNodes) {
        			node.updateWeight(learningRate);
        		}
        		

        		
    			//System.out.println("LossAfter: " + String.format("%.3e",lossAfter));
    		}
    		totalLoss = 0;
    		for(Instance inst : trainingSet){
    			double lossAfter = -loss(inst);
    			totalLoss += lossAfter;
    		}
    		totalLoss = totalLoss/trainingSet.size();
	  		System.out.println("Epoch: " + epoch + ", Loss: " + String.format("%.3e", totalLoss));
    	}
    }

    /**
     * Calculate the cross entropy loss from the neural network for
     * a single instance.
     * The parameter is a single instance
     */
    private double loss(Instance instance) {
    	
		int i = 0;
    	for(Double input: instance.attributes) {
    		inputNodes.get(i).setInput(input);
    		i++;
    	}
    	
    	for(Node node : hiddenNodes) {
			node.calculateOutput();
		}
		
		double sum= 0;
		for(Node node : outputNodes) {
			node.calculateOutput();
			sum += node.getOutput();
		}
		for(Node node : outputNodes) {
			node.calculateOutput(sum);
		}
		double loss = 0;
		int index = 0;
		for(Node node : outputNodes) {
			loss += instance.classValues.get(index)*Math.log(node.getOutput());
			index++;
		}
        return loss;
    }
}
