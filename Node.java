import java.util.*;

/**
 * Class for internal organization of a Neural Network.
 * There are 5 types of nodes. Check the type attribute of the node for details.
 * Feel free to modify the provided function signatures to fit your own implementation
 */

public class Node {
    private int type = 0; //0=input,1=biasToHidden,2=hidden,3=biasToOutput,4=Output
    public ArrayList<NodeWeightPair> parents = null; //Array List that will contain the parents (including the bias node) with weights if applicable

    private double inputValue = 0.0;
    private double outputValue = 0.0;
    private double outputGradient = 0.0;
    private double delta = 0.0; //input gradient

    //Create a node with a specific type
    Node(int type) {
        if (type > 4 || type < 0) {
            System.out.println("Incorrect value for node type");
            System.exit(1);

        } else {
            this.type = type;
        }

        if (type == 2 || type == 4) {
            parents = new ArrayList<>();
        }
    }

    //For an input node sets the input value which will be the value of a particular attribute
    public void setInput(double inputValue) {
        if (type == 0) {    //If input node
            this.inputValue = inputValue;
        }
    }

    /**
     * Calculate the output of a node.
     * You can get this value by using getOutput()
     */
    public void calculateOutput() {
        if (type == 2 || type == 4) {//Not an input or bias node
        	double z = 0;
        	
        	for(NodeWeightPair nwp : this.parents){
        		z += nwp.weight*nwp.node.getOutput();
        	}
        	
        	if(type ==2) { //ReLU
        		if(z <= 0){
        			this.outputValue = 0;
        		}else{
        			this.outputValue = z;
        		}
        		
        	}else if(type == 4){ //softimax
        		this.outputValue = Math.exp(z);
        	}
            // TODO: add code here
        }
        
    }
    public void calculateOutput(double sum) {
    	this.outputValue = this.outputValue/sum;
    }

    //Gets the output value
    public double getOutput() {

        if (type == 0) {    //Input node
            return inputValue;
        } else if (type == 1 || type == 3) {    //Bias node
            return 1.00;
        } else {
            return outputValue;
        }

    }

    //Calculate the delta value of a node.
    public void calculateDelta( ArrayList<Node> outputNodes, int y) {
        if (type == 2 || type == 4)  {
        	if(type ==4) { //output
        		this.delta = y - this.outputValue;
        		//y_j - g(z_j)
        	}else {		//hidden
        		
        		if(this.outputValue <= 0){
        			this.delta = 0;
        		}else{
        			double sum = 0;
            		for(Node node : outputNodes) {
            			for(NodeWeightPair nwp : node.parents) {
                			if(this.equals(nwp.node)) {
                				sum += nwp.weight*node.delta;
                			}
            			}
            		}
            		this.delta = sum;
        		}
        	}
        }
    }


    //Update the weights between parents node and current node
    public void updateWeight(double learningRate) {
        if (type == 2 || type == 4) {
        	for(NodeWeightPair nwp : this.parents) {
        		nwp.weight = nwp.weight + (learningRate*nwp.node.getOutput()*this.delta);
        	}
            // TODO: add code here
        }
    }
}


