%{

Script: MultilayerPerceptron.m
Version of the MATLAB implemented: 2018a.

Author: Nadia Matos
email: nadiamatos.18@gmail.com

This script is of Multilayer Perceptron RNA, doing learning and
operation process. Using algorithm back-propagation.

%}
classdef MultilayerPerceptron < handle

  properties

    errorMeanSquareCurrent = 0; errorMeanSquareLast = 0;
    quantitySampleTraining = 90; % in percentage
    quantityOutputRna = 1; quantityInputRna = 0;
    inputRna = []; outputRnaDesired = []; quantitySample = 0;
    biasRna = cell(1); weigthRna = cell(1); rateLearning = 0.05; epsilon = 10e-10; epoch = 0;
    layoutRna = []; % [quantInputs, quantNeuronsHiddenLayer1, ..., quantNeuronsHiddenLayerN, quantOutputs]
    quantityLayerHidden = 0; quantityLayer = 0;
    errorSquare = 0;

  end

  methods

    function obj = MultilayerPerceptron(database, percentageForTraining, layoutRna, quantInputs)

      obj.layoutRna = layoutRna;
      obj.quantityLayer = size(layoutRna, 2);
      obj.quantityLayerHidden = obj.quantityLayer-2;
      obj.quantityOutputRna = obj.layoutRna(end);
      obj.quantityInputRna = obj.layoutRna(end)+1;
      obj.inputRna = [ones(size(database, 1), 1), table2array(database(:, 1:quantInputs))];
      obj.outputRnaDesired = table2array(database(:, end));
      obj.quantitySample = size(database, 1);
      obj.quantitySampleTraining = floor(size(database, 1)*percentageForTraining/100);
      % obj.inputRna = obj.normalization(obj.inputRna);

    end

    function calculateWeigths(obj)

      %{
        This function calculate the weigths of the neural network,
        with bias too.

        obj.weigthRna{:, :, layer}(neuron layer-m, :)
      %}      

      % k corresponde as camadas. 

      for k = 1 : obj.quantityLayer-1
        %obj.biasRna{:, :, k}   = rand(obj.layoutRna(k+1), 1, 1);        
        obj.weigthRna{:, :, k} = rand(obj.layoutRna(k+1), obj.layoutRna(k)+1, 1);        
      end

    end

    function data = normalization(obj, data, c)

      for i = 2 : 1 : size(data, 2)
        data(:, i) = (data(:, i) - min(obj.inputRna(:, i)))./(max(obj.inputRna(:, i)) - min(obj.inputRna(:, i)));
      end

    end

    function e = calculateErrorSquare(obj, erro)

      e = 0;
      for k = 1 : size(erro, 1)
        e = e + erro(k)*erro(k);
      end

    end

    function out = logistic(obj, u)

      a = 1;

      out = 1./( 1 + exp(-a*u) );

    end

    function out = logisticDerivate(obj, u)

      a = 1;

      out = ( a*exp(-a*u) )./( (1 + exp(-a*u)).^2 );

    end


    function training(obj)

      difference = inf; errorSquare = 0; aux = 0;
      y = cell(1); u = cell(1);
      d = obj.outputRnaDesired; e = zeros(size(d, 1), 1);
      errorMeanSquareLast = 0; errorMeanSquareCurrent = 0;

      while (difference >= obj.epsilon)

        disp('Training ...');

        %for k = 1 : obj.quantitySampleTraining

        for k = 1 : 1

          % forward
          for j = 1 : obj.quantityLayer-1

            if (isequal(j, 1)); aux = obj.inputRna(k, :)';
            else                aux = [1; y{:, :, j-1}]; end                    

            u{:, :, j} = (obj.weigthRna{:, :, j}*aux);
            y{:, :, j} = obj.logistic( u{:, :, j} );


            %{
            disp('----------------------------');
            fprintf('j = %d\n', j);            
            disp('aux');
            disp(aux);
            disp('obj.weigthRna{:, :, j}');
            disp(obj.weigthRna{:, :, j});
            disp('u{:, :, j}');
            disp(u{:, :, j});   
            disp('y{:, :, j}');
            disp(y{:, :, j});           
            %}

          end

          %
          e = d(k) - y{:, :, end}';
          obj.errorSquare = obj.errorSquare + obj.calculateErrorSquare(e);

          % backward
          for l = obj.quantityLayer : -1 : 2
            if (isequal(l, obj.quantityLayer))

              epsilon{:, :, l-1} = e*obj.logisticDerivate([u{:, :, end}]);
              obj.weigthRna{:, :, l-1} = obj.weigthRna{:, :, l-1} + obj.rateLearning*(epsilon{:, :, l-1}.*y{:, :, l-1});

            else

              %epsilon{:, :, l-1} = obj.logisticDerivate(v{:, :, l-2}).*(epsilon{:, :, l}*obj.weigthRna{:, :, l}');
              epsilon{:, :, l-1} = obj.logisticDerivate(u{:, :, l})*(epsilon{:, :, l}.*obj.weigthRna{:, :, l}(:, 2:end)');
              obj.weigthRna{:, :, l-1} = obj.weigthRna{:, :, l-1} + obj.rateLearning*(epsilon{:, :, l-1}.*y{:, :, l-1}(2:end))';

            end

            %obj.weigthRna{:, :, l-1} = obj.weigthRna{:, :, l-1} + obj.rateLearning*(epsilon{:, :, l-1}.*y{:, :, l-1});

          end
          %}
        end        
      end

      obj.epoch = obj.epoch + 1;
      obj.errorMeanSquareLast = obj.errorMeanSquareCurrent;
      obj.errorMeanSquareCurrent = obj.errorSquare/(2*obj.quantitySampleTraining);
      difference = abs(obj.errorMeanSquareCurrent - obj.errorMeanSquareLast);
      

    end

    function saveWeigths(obj)
      % save data of weigths of mlp in a file
    end

    function readWeigths(obj)
      % read data of weigths of mlp in a file
    end

    function operation(obj)

    end
  end

end
