#include "tinn.h"
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

// Data object.
typedef struct
{
    // 2D floating point array of input.
    float** in;
    // 2D floating point array of target.
    float** tg;
    // Number of inputs to neural network.
    int nips;
    // Number of outputs to neural network.
    int nops;
    // Number of rows in file (number of sets for neural network).
    int rows;
}
Data;

// Returns the number of lines in a file.
static int lns(FILE*  file)
{
    int ch = EOF;
    int lines = 0;
    int pc = '\n';
    while((ch = getc(file)) != EOF)
    {
        if(ch == '\n')
            lines++;
        pc = ch;
    }
    if(pc != '\n')
        lines++;
    rewind(file);
    return lines;
}

// Reads a line from a file.
static char* readln(FILE*  file)
{
    int ch = EOF;
    int reads = 0;
    int size = 128;
    char* line = (char*) malloc((size) * sizeof(char));
    while((ch = getc(file)) != '\n' && ch != EOF)
    {
        line[reads++] = ch;
        if(reads + 1 == size)
            line = (char*) realloc((line), (size *= 2) * sizeof(char));
    }
    line[reads] = '\0';
    return line;
}

// New 2D array of floats.
static float** new2d( int rows,  int cols)
{
    float** row = (float**) malloc((rows) * sizeof(float*));
    for(int r = 0; r < rows; r++)
        row[r] = (float*) malloc((cols) * sizeof(float));
    return row;
}

// New data object.
static Data ndata( int nips,  int nops,  int rows)
{
     Data data = {
        new2d(rows, nips), new2d(rows, nops), nips, nops, rows
    };
    return data;
}

// Gets one row of inputs and outputs from a string.
static void parse( Data data, char* line,  int row)
{
     int cols = data.nips + data.nops;
    for(int col = 0; col < cols; col++)
    {
         float val = atof(strtok(col == 0 ? line : NULL, " "));
        if(col < data.nips)
            data.in[row][col] = val;
        else
            data.tg[row][col - data.nips] = val;
    }
}

// Frees a data object from the heap.
static void dfree( Data d)
{
    for(int row = 0; row < d.rows; row++)
    {
        free(d.in[row]);
        free(d.tg[row]);
    }
    free(d.in);
    free(d.tg);
}


// Parses file from path getting all inputs and outputs for the neural network. Returns data object.
static Data build( char* path,  int nips,  int nops)
{
    FILE* file = fopen(path, "r");
    if(file == NULL)
    {
        printf("Could not open %s\n", path);
        printf("Get it from the machine learning database: ");
        printf("wget http://archive.ics.uci.edu/ml/machine-learning-databases/semeion/semeion.data\n");
        exit(1);
    }
     int rows = lns(file);
    Data data = ndata(nips, nops, rows);
    for(int row = 0; row < rows; row++)
    {
        char* line = readln(file);
        parse(data, line, row);
        free(line);
    }
    fclose(file);
    return data;
}

// Learns and predicts hand written digits with 98% accuracy.
int main()
{
    // Tinn does not seed the random number generator.
    srand(time(0));
    // Input and output size is harded coded here as machine learning
    // repositories usually don't include the input and output size in the data itself.
     int nips = 784;
     int nops = 10;
    // Hyper Parameters.
    // Learning rate is annealed and thus not ant.
    // It can be fine tuned along with the number of hidden layers.
    // Feel free to modify the anneal rate.
    // The number of iterations can be changed for stronger training.
     //float rate = 1.0f;
     //int nhid = 28;
     //float anneal = 0.99f;
     //int iterations = 1000;
     float* in;
     float* tg;
     float* pd;
     int b; 
     int i;

    // Load the training set.
     Data data = build("mnist.txt", nips, nops);
    // Train, baby, train.
   // This is how you save the neural network to disk.
    // xtsave(tinn, "saved.tinn");
    // This is how you load the neural network from disk.
     Tinn loaded = xtload("saved.tinn");
    // Now we do a prediction with the neural network we loaded from disk.
    // Ideally, we would also load a testing set to make the prediction with,
    // but for the sake of brevity here we just reuse the training set from earlier.
    // One data set is picked at random (zero index of input and target arrays is enough
    // as they were both shuffled earlier).
    while (1)
    {
     
         printf("==========================================\r\n");
         b = rand() % data.rows;
         printf("random select row %d\r\n",b);
         in = data.in[b];
         tg = data.tg[b];
         
         for(i=0; i<data.nips; i++)
         {
             printf("%2x",(int)(in[i]*255));
             if ((i % 28) == 0)
                printf("\r\n");
         }        
       
         printf("\r\n----------------\r\n");


         pd = xtpredict(loaded, in);
        // Prints target.
         xtprint(tg, data.nops);
        // Prints prediction.
        xtprint(pd, data.nops);
        getchar();

     }
    // All done. Let's clean up.
    xtfree(loaded);
    dfree(data);
    return 0;
}
