CXX = clang++
CXXFLAGS = -std=c++17 -Wall -Wextra -O2 -fPIC -I/opt/homebrew/include 
LDFLAGS = -L/opt/homebrew/lib 
LIBS = -lsymengine -lflint -lgmp -lmpfr

TARGET = main
SRC = $(wildcard *.cpp)
OBJ = $(SRC:.cpp=.o)
SOURCES = main.cpp 
OBJECTS = $(SOURCES:.cpp=.o)

all: $(TARGET)

$(TARGET): $(OBJ)
	$(CXX) $(CXXFLAGS) $(LDFLAGS) $(OBJ) -o $(TARGET) $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -f $(OBJ) $(TARGET)

rebuild: clean all

run: $(TARGET)
	./$(TARGET)

.PHONY: all clean rebuild run
