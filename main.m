%{

Script: main.m
Version of the MATLAB implemented: 2018a.

Author: Nadia Matos
email: nadiamatos.18@gmail.com

This script call the MultilayerPerceptron.m script.

%}

clc; clear ('all'); close all;
format short;

database = readtable('database-xor.txt');
quantityinputs = size(database, 2)-1;

% [quantInputs, quantNeuronsHiddenLayer1, ..., quantNeuronsHiddenLayerN, quantOutputs]
% [2 2 1] -> 2 entradas, 1 primeira camada escondida de 2 neuronios, 1 saida.
layoutRna = [2 3 2 1];

rna = MultilayerPerceptron(database, 90, layoutRna, quantityinputs);

rna.calculateWeigths();
rna.training();
%rna.operation();
