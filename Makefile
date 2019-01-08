BIN1 = train
BIN2 = predict

CFLAGS = -Wall
#CFLAGS =  -Wall -Wextra -pedantic -Ofast -flto -march=native

LDFLAGS =  -L/lib -L/usr/lib

CC = gcc

SRC1 = train.c tinn.c
SRC2 = predict.c tinn.c

all:
	$(CC) $(CFLAGS) $(LDFLAGS) $(SRC1) -o $(BIN1) -lm
	$(CC) $(CFLAGS) $(LDFLAGS) $(SRC2) -o $(BIN2) -lm

run1:
	./$(BIN1)
run2:
	./$(BIN2)


clean:
	rm -f $(BIN1);  rm -f $(BIN2)

