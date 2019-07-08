/*
 * Efe Önal
 * 2016400267
 * Compiling
 * Working
 */

#include <mpi.h>
#include <iostream>
#include <vector>
#include <sstream>
#include <fstream>
#include <iterator>
#include <stdio.h>
#include <cstring>
#include <string>
#include <cmath>

using namespace std;

template <class Container>

vector<string> split(string str, char delimiter) {
    vector<string> internal;
    stringstream ss(str); // Turn the string into a stream.
    string tok;

    while(getline(ss, tok, delimiter)) {
        internal.push_back(tok);
    }

    return internal;
}


const int NUM_WORDS = 1000; //There will always be 1000 words in the matrix.
const int MAX_LINE_LEN = 6000;
const int MAX_WORD_LEN = 50;
const int EMBEDDING_DIMENSION = 300;
const char DELIMITER[2] = "\t";
const int COMMAND_EXIT = 0;
const int COMMAND_QUERY = 1;
const int COMMAND_CALCULATE_SIMILARITY = 2;
const int COMMAND_WORD_NOT_FOUND = 3;

//In this function wedistribute the words and their vectors to processes.
void distributeEmbeddings(char *filename, int numberOfSlaves) {


    int artan = NUM_WORDS % numberOfSlaves; //number of words that are left after each processor gets the same number of words.
    int evenNumberLines = NUM_WORDS / numberOfSlaves;   //Number of lines each slave is going to have.
    int linePivot = 0;

    char line[MAX_LINE_LEN];
    FILE *file = fopen(filename, "r");
    int wordIndex = 0;
    int p = 1;

    //First artan processors will get one line more than the others.
    for (int p = 0; p < artan; p++) {
        float *embeddings_matrix = (float *) malloc(sizeof(float) * (evenNumberLines+1) * EMBEDDING_DIMENSION);

        char *words = (char *) malloc(sizeof(char) * (evenNumberLines+1) * MAX_WORD_LEN);
        //We fill words and embedding_matrix.
        for(int i = 0; i < evenNumberLines+1; i++){
            fgets(line, MAX_LINE_LEN, file);
            linePivot++;
            char *word;
            word = strtok(line, DELIMITER);
            strcpy(words + i * MAX_WORD_LEN, word);
            for (int embIndex = 0; embIndex < EMBEDDING_DIMENSION; embIndex++) {
                char *field = strtok(NULL, DELIMITER);
                float emb = strtof(field, NULL);
                *(embeddings_matrix + i * EMBEDDING_DIMENSION + embIndex) = emb;
            }
        }
        //We send words, embedding_matrix, and the overall index of the first line of processes' data.
        int indexToSend = linePivot - evenNumberLines - 1;
        MPI_Send(
                /* data         = */ words,
                /* count        = */ (evenNumberLines+1)*MAX_WORD_LEN,
                /* datatype     = */ MPI_CHAR,
                /* destination  = */ p+1,
                /* tag          = */ 0,
                /* communicator = */ MPI_COMM_WORLD);
        MPI_Send(
                /* data         = */ embeddings_matrix,
                /* count        = */ (evenNumberLines+1)*EMBEDDING_DIMENSION,
                /* datatype     = */ MPI_FLOAT,
                /* destination  = */ p+1,
                /* tag          = */ 0,
                /* communicator = */ MPI_COMM_WORLD);
        MPI_Send(
                /* data         = */ &indexToSend,
                /* count        = */ sizeof(int),
                /* datatype     = */ MPI_BYTE,
                /* destination  = */ p+1,
                /* tag          = */ 0,
                /* communicator = */ MPI_COMM_WORLD);

        //We delete embedding_matrix and words.
        free(embeddings_matrix);
        free(words);
    }
    //The rest of the processes get the same number of lines.
    for(int p = artan; p < numberOfSlaves; p++){
        float *embeddings_matrix = (float *) malloc(sizeof(float) * (evenNumberLines) * EMBEDDING_DIMENSION);

        char *words = (char *) malloc(sizeof(char) * (evenNumberLines) * MAX_WORD_LEN);
        for(int i = 0; i < evenNumberLines; i++){
            fgets(line, MAX_LINE_LEN, file);
            linePivot++;
            char *word;
            word = strtok(line, DELIMITER);
            strcpy(words + i * MAX_WORD_LEN, word);
            for (int embIndex = 0; embIndex < EMBEDDING_DIMENSION; embIndex++) {
                char *field = strtok(NULL, DELIMITER);
                float emb = strtof(field, NULL);
                *(embeddings_matrix + i * EMBEDDING_DIMENSION + embIndex) = emb;
            }
        }
        int indexToSend2 = linePivot - evenNumberLines - 1;
        MPI_Send(
                /* data         = */ words,
                /* count        = */ evenNumberLines*MAX_WORD_LEN,
                /* datatype     = */ MPI_CHAR,
                /* destination  = */ p+1,
                /* tag          = */ 0,
                /* communicator = */ MPI_COMM_WORLD);
        MPI_Send(
                /* data         = */ embeddings_matrix,
                /* count        = */ evenNumberLines*EMBEDDING_DIMENSION,
                /* datatype     = */ MPI_FLOAT,
                /* destination  = */ p+1,
                /* tag          = */ 0,
                /* communicator = */ MPI_COMM_WORLD);
        MPI_Send(
                /* data         = */ &indexToSend2,
                /* count        = */ sizeof(int),
                /* datatype     = */ MPI_BYTE,
                /* destination  = */ p+1,
                /* tag          = */ 0,
                /* communicator = */ MPI_COMM_WORLD);
        free(embeddings_matrix);
        free(words);
    }
}

//This is the code for the master node.
void runMasterNode(int world_rank, int numberOfSlaves) {
    //We first distribute embeddings to slaves.
    distributeEmbeddings("./word_embeddings_1000.txt", numberOfSlaves);
    //This while loop lets the program take  multiple inputs from the user.
    while (true) {
        char queryWord[256];
        cout << "Please type a query word:";
        cin >> queryWord;

        //If user types EXIT, master broadcasts 0, telling the slaves to exit.
        //And master brakes the while loop to exit.
        if(strcmp(queryWord, "EXIT") == 0){
            MPI_Bcast(
                    (void*) &COMMAND_EXIT,
                    1,
                    MPI_INT,
                    0,
                    MPI_COMM_WORLD
                    );
            break;
        }
        int *indexes;
        int a;
        bool found = false;
        int foundIndex;
        int finderProcessor;
        float *matrixRecieved = (float *) malloc(sizeof(float) * EMBEDDING_DIMENSION);
        float *similarities = (float *) malloc(sizeof(float) * (numberOfSlaves + 1));
        char *similarWords = (char *) malloc(sizeof(char) * (numberOfSlaves + 1) * MAX_WORD_LEN);
        char *bos = (char *) malloc(sizeof(char) * MAX_WORD_LEN);
        float *bosSimilarity = (float *) malloc(sizeof(float));

        //Master broadcasts there's a word to search for.
        MPI_Bcast(
                (void *) &COMMAND_QUERY,
                1,
                MPI_INT,
                0,
                MPI_COMM_WORLD);

        //Master broadcasts the word to search.
        MPI_Bcast(
                (void *) queryWord,
                MAX_WORD_LEN,
                MPI_CHAR,
                0,
                MPI_COMM_WORLD);
        indexes = (int *) malloc(sizeof(int) * (numberOfSlaves + 1));

        //Master gathers the search results from the slaves. The result is -1 if word is not found, index of the word if it is found.
        MPI_Gather(&a, 1, MPI_INT, indexes, 1, MPI_INT, 0, MPI_COMM_WORLD);
        //It the word exists, found is true and finder processor is known.
        for (int i = 1; i < numberOfSlaves + 1; i++) {
            if ((indexes[i]) != -1) {
                foundIndex = indexes[i];
                finderProcessor = i;
                found = true;
            }
        }

        //If the word exists;
        if (found) {
            //Master recieves vector of the word.
            MPI_Recv(matrixRecieved, EMBEDDING_DIMENSION, MPI_FLOAT, finderProcessor, 0, MPI_COMM_WORLD,
                     MPI_STATUS_IGNORE);
            //Then master broadcasts COMMAND_CALCULATE_SIMILARITY, which is 2.
            MPI_Bcast(
                    (void *) &COMMAND_CALCULATE_SIMILARITY,
                    1,
                    MPI_INT,
                    0,
                    MPI_COMM_WORLD);
            //Then master broadcasts the vector of user's choice of word.
            MPI_Bcast(
                    matrixRecieved,
                    EMBEDDING_DIMENSION,
                    MPI_FLOAT,
                    0,
                    MPI_COMM_WORLD
            );
            //Master gathers the most similar words from the slaves.
            MPI_Gather(bos, MAX_WORD_LEN, MPI_CHAR, similarWords, MAX_WORD_LEN, MPI_CHAR, 0, MPI_COMM_WORLD);
            //Master gathers vectors of the most similar words from the slaves.
            MPI_Gather(bosSimilarity, 1, MPI_FLOAT, similarities, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
            for (int l = 1; l < numberOfSlaves + 1; l++) {
                cout << "***** word: " << similarWords + l * MAX_WORD_LEN << ", similarity: " << similarities[l]
                     << endl;
            }
        } else {
            //If the word is not found in  our data;
            //COMMAND_WORD_NOT_FOUND is broadcasted to the slaves.
            cout << "Query word was not found" << endl;
            MPI_Bcast(
                    (void *) &COMMAND_WORD_NOT_FOUND,
                    1,
                    MPI_INT,
                    0,
                    MPI_COMM_WORLD);
        }
        found = false;
    }
}




void runSlaveNode(int world_rank, int numberOfSlaves){
    bool artanProcessor;
    int artan = NUM_WORDS % numberOfSlaves;
    int evenNumberLines = NUM_WORDS / numberOfSlaves;
    int indexRecieved;
    int command;
    char queryWord[256];
    bool wordFound= false;
    char returnWord[256];
    int indexFound = -1;
    float toplam = 0;
    float boy1 = 0;
    float boy2 = 0;
    float reelBoy = 0;
    float similarity = 0;
    float max = 0;
    int indexOfMax = 0;

    //If word rank  is less than extra number of lines, the processor is an artanProcessor.
    if(world_rank <= (artan + 1)){
        artanProcessor = true;
    }else{
        artanProcessor  = false;
    }
    char* words = (char*)malloc(sizeof(char) * (evenNumberLines+1)*MAX_WORD_LEN);
    char* words1 = (char*)malloc(sizeof(char) * (evenNumberLines)*MAX_WORD_LEN);
    float* embeddings_matrix = (float*)malloc(sizeof(float) * (evenNumberLines+1)*EMBEDDING_DIMENSION);
    float* embeddings_matrix1 = (float*)malloc(sizeof(float) * (evenNumberLines)*EMBEDDING_DIMENSION);

    //If it's an artanProcessor, it revieves an extra word and an extra row of floats.
    if(artanProcessor){
        MPI_Recv(words, (evenNumberLines+1)*MAX_WORD_LEN, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(embeddings_matrix, (evenNumberLines+1)*EMBEDDING_DIMENSION, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(&indexRecieved, sizeof(int), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    } else{
        MPI_Recv(words1, (evenNumberLines)*MAX_WORD_LEN, MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(embeddings_matrix1, (evenNumberLines)*EMBEDDING_DIMENSION, MPI_FLOAT, 0, 0, MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Recv(&indexRecieved, sizeof(int), MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    //This while loop enables the program to accept multiple input.
    while(true) {
        float* similarities = (float*)malloc(sizeof(float) * (evenNumberLines+1));
        float* similarities1 = (float*)malloc(sizeof(float) * (evenNumberLines));

        float* matrixToSend = (float*)malloc(sizeof(float) *EMBEDDING_DIMENSION);
        float* matrixToCompareWith = (float*)malloc(sizeof(float) *EMBEDDING_DIMENSION);

        //Slaves recieve a command.
        MPI_Bcast(
                (void *) &command,
                1,
                MPI_INT,
                0,
                MPI_COMM_WORLD
        );

        //If the command is one, COMMAND_QUERY, it gets the query word from the master via MPI_Bcast.
        if (command == 1) {
            MPI_Bcast(
                    (void *) queryWord,
                    MAX_WORD_LEN,
                    MPI_CHAR,
                    0,
                    MPI_COMM_WORLD);
            //After taking the query word, if the processor is an artanProcessor for loop are going to execute number of lines
            //each processor recieves plus 1 times. Else, they are going to execute for even numbeer of lines times.
            if (artanProcessor) {
                //For all lines of the processor;
                for (int i = 0; i < evenNumberLines + 1; i++) {
                    //If the word is same as the queryWord, word is found, we know at what index.
                    if (strcmp(words + i * MAX_WORD_LEN, queryWord) == 0) {
                        wordFound = true;
                        indexFound = i;
                    }
                }
            } else {
                for (int i = 0; i < evenNumberLines; i++) {
                    if (strcmp(words1 + i * MAX_WORD_LEN, queryWord) == 0) {
                        wordFound = true;
                        indexFound = i;
                    }
                }
            }
            //Slaves send indexFound values to master processor via MPI_Gather.
            //indexFound is the index if the word is found, else, it is -1.
            MPI_Gather(&indexFound, 1, MPI_INT, NULL, 1, MPI_INT, 0, MPI_COMM_WORLD);

            //If the word is found in a slave.
            if (wordFound) {
                //Processor prepares the embeddding values of the word in matrixToSend.
                if (artanProcessor) {
                    for (int i = 0; i < EMBEDDING_DIMENSION; i++) {
                        matrixToSend[i] = embeddings_matrix[indexFound * EMBEDDING_DIMENSION + i];
                    }
                } else {
                    for (int i = 0; i < EMBEDDING_DIMENSION; i++) {
                        matrixToSend[i] = embeddings_matrix1[indexFound * EMBEDDING_DIMENSION + i];
                    }
                }
                //The finderProcessor sends the values to the master.
                MPI_Send(
                        /* data         = */ matrixToSend,
                        /* count        = */ EMBEDDING_DIMENSION,
                        /* datatype     = */ MPI_FLOAT,
                        /* destination  = */ 0,
                        /* tag          = */ 0,
                        /* communicator = */ MPI_COMM_WORLD);
            }
        }
        //else if command = 0, EXIT, break the while loop so you will exit.
        else if(command == 0){
            break;
        }
        //Every slaves recieves a command from the master process.
        MPI_Bcast(
                (void *) &command,
                1,
                MPI_INT,
                0,
                MPI_COMM_WORLD
        );
        //If the command is CALCULATE_SIMILARITY,
        if (command == 2) {
            //Slaves recieve the matrix of the query word.
            MPI_Bcast(
                    matrixToCompareWith,
                    EMBEDDING_DIMENSION,
                    MPI_FLOAT,
                    0,
                    MPI_COMM_WORLD
            );

            //If the processor is an artanProceessor;
            if (artanProcessor) {
                //This for loop calculats the similarities and puts them into similarities.
                for (int j = 0; j < evenNumberLines + 1; j++) {
                    for (int i = 0; i < EMBEDDING_DIMENSION; i++) {
                        toplam = toplam + (matrixToCompareWith[i] * embeddings_matrix[(EMBEDDING_DIMENSION * j) + i]);
                        boy1 = boy1 + matrixToCompareWith[i] * matrixToCompareWith[i];
                        boy2 = boy2 + embeddings_matrix[(EMBEDDING_DIMENSION * j) + i] *
                                      embeddings_matrix[(EMBEDDING_DIMENSION * j) + i];
                    }
                    boy1 = sqrt(boy1);
                    boy2 = sqrt(boy2);
                    reelBoy = boy1 * boy2;
                    similarity = toplam / reelBoy;
                    similarities[j] = similarity;
                    toplam = 0;
                    boy1 = 0;
                    boy2 = 0;
                    reelBoy = 0;
                    similarity = 0;
                }

                //index of the maximum similar word is known, value of max similarity is known.
                for (int i = 0; i < evenNumberLines + 1; i++) {
                    if (similarities[i] > max) {
                        max = similarities[i];
                        indexOfMax = i;
                    }
                }
                //Slaves send the most similar words to master process.
                MPI_Gather(words + indexOfMax * MAX_WORD_LEN, MAX_WORD_LEN, MPI_CHAR, NULL, MAX_WORD_LEN, MPI_CHAR, 0,
                           MPI_COMM_WORLD);
                //Slaves send maximum similarity values to master process.
                MPI_Gather(&max, 1, MPI_FLOAT, NULL, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
            } else {
                for (int j = 0; j < evenNumberLines; j++) {
                    for (int i = 0; i < EMBEDDING_DIMENSION; i++) {
                        toplam = toplam + (matrixToCompareWith[i] * embeddings_matrix1[(EMBEDDING_DIMENSION * j) + i]);
                        boy1 = boy1 + matrixToCompareWith[i] * matrixToCompareWith[i];
                        boy2 = boy2 + embeddings_matrix1[(EMBEDDING_DIMENSION * j) + i] *
                                      embeddings_matrix1[(EMBEDDING_DIMENSION * j) + i];
                    }
                    boy1 = sqrt(boy1);
                    boy2 = sqrt(boy2);
                    reelBoy = boy1 * boy2;
                    similarity = toplam / reelBoy;
                    similarities1[j] = similarity;
                    toplam = 0;
                    boy1 = 0;
                    boy2 = 0;
                    reelBoy = 0;
                    similarity = 0;
                }
                for (int i = 0; i < evenNumberLines; i++) {
                    if (similarities1[i] > max) {
                        max = similarities1[i];
                        indexOfMax = i;
                    }
                }
                MPI_Gather(words1 + indexOfMax * MAX_WORD_LEN, MAX_WORD_LEN, MPI_CHAR, NULL, MAX_WORD_LEN, MPI_CHAR, 0,
                           MPI_COMM_WORLD);
                MPI_Gather(&max, 1, MPI_FLOAT, NULL, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);
            }
        } else if (command == 3) {
            //else if command is  COMMAND_WORD_NOT_FOUND do nothing.
        }
        free(similarities);
        free(similarities1);
        free(matrixToCompareWith);
        free(matrixToSend);
        indexFound = -1;
        toplam = 0;
        boy1 = 0;
        boy2 = 0;
        reelBoy = 0;
        similarity = 0;
        max = 0;
        indexOfMax = 0;
        wordFound = false;
    }
    free(words);
    free(words1);
    free(embeddings_matrix);
    free(embeddings_matrix1);
}

int main(int argc, char** argv){
    MPI_Init(NULL, NULL);       //Initialize MPI environment.

    //Get the number of processes.
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //Get the rank of the processes.
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);


    //Assuming there are at least two processors; one master, one slave
    if(world_size < 2){
        cout << "World size must be greater than 1" << endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    int wordIndex;
    int numberOfSlaves = world_size - 1;
    if(world_rank == 0){
        runMasterNode(world_rank, numberOfSlaves);
    }
    else{
        runSlaveNode(world_rank, numberOfSlaves);
    }
    MPI_Finalize();
    //cout << "Process " << world_rank << " stopped." << endl;
}