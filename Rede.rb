require 'matrix'
require 'gnuplot'

# Função para gerar dados pluviométricos simulados
def generate_rainfall_data(size)
  data = []
  size.times do |i|
    rainfall = 100 * Math.sin(i * 0.1) + rand(20)
    data << rainfall
  end
  data
end

# Normalização dos dados
def normalize(data)
  min = data.min
  max = data.max
  data.map { |x| (x - min) / (max - min) }
end

def denormalize(data, min, max)
  data.map { |x| x * (max - min) + min }
end

# Preparação dos dados
def prepare_data(data, window_size)
  x, y = [], []
  (0..data.size - window_size - 1).each do |i|
    x << data[i, window_size]
    y << data[i + window_size]
  end
  [x, y]
end

# Classe da Rede Neural
class NeuralNetwork
  attr_accessor :input_size, :hidden_size, :output_size, :learning_rate, :weights_input_hidden, :weights_hidden_output

  def initialize(input_size, hidden_size, output_size, learning_rate)
    @input_size = input_size
    @hidden_size = hidden_size
    @output_size = output_size
    @learning_rate = learning_rate
    @weights_input_hidden = Matrix.build(@hidden_size, @input_size) { randn * 0.01 }
    @weights_hidden_output = Matrix.build(@output_size, @hidden_size) { randn * 0.01 }
  end

  def relu(x)
    x.map { |v| [0, v].max }
  end

  def relu_derivative(x)
    x.map { |v| v > 0 ? 1 : 0 }
  end

  def train(x, y)
    # Forward pass
    input = Matrix[x]
    hidden_input = @weights_input_hidden * input.transpose
    hidden_output = relu(hidden_input)
    final_input = @weights_hidden_output * hidden_output
    final_output = relu(final_input)

    # Calculate error
    target = Matrix[[y]]
    output_error = target - final_output
    hidden_error = @weights_hidden_output.transpose * output_error

    # Backpropagation
    delta_output = Matrix.build(@output_size, 1) { |row, col| output_error[row, col] * relu_derivative(final_output)[row, col] }
    delta_hidden = Matrix.build(@hidden_size, 1) { |row, col| hidden_error[row, col] * relu_derivative(hidden_output)[row, col] }

    @weights_hidden_output += delta_output * hidden_output.transpose * @learning_rate
    @weights_input_hidden += delta_hidden * input * @learning_rate

    output_error
  end

  def predict(x)
    input = Matrix[x]
    hidden_input = @weights_input_hidden * input.transpose
    hidden_output = relu(hidden_input)
    final_input = @weights_hidden_output * hidden_output
    final_output = relu(final_input)
    final_output.to_a.flatten
  end

  private

  def randn
    Math.sqrt(-2 * Math.log(rand)) * Math.cos(2 * Math::PI * rand)
  end
end

# Função para calcular MAE
def mean_absolute_error(y_true, y_pred)
  y_true.zip(y_pred).map { |t, p| (t - p).abs }.sum / y_true.size.to_f
end

# Função para calcular MSE
def mean_squared_error(y_true, y_pred)
  y_true.zip(y_pred).map { |t, p| (t - p)**2 }.sum / y_true.size.to_f
end

# Configuração e treinamento da rede neural
rainfall_data = generate_rainfall_data(360) # 30 anos de dados mensais
rainfall_data_normalized = normalize(rainfall_data)
x_normalized, y_normalized = prepare_data(rainfall_data_normalized, window_size = 5)

input_size = window_size
hidden_size = 20 # Aumentar o número de neurônios na camada oculta
output_size = 1
learning_rate = 0.1

nn = NeuralNetwork.new(input_size, hidden_size, output_size, learning_rate)

# Dados de teste
test_data = generate_rainfall_data(12) # Teste com 1 ano de dados mensais
test_data_normalized = normalize(test_data)
test_inputs_normalized = test_data_normalized.each_cons(window_size).to_a
real_values = [62, 65, 63.5, 73.5, 79, 97, 105, 110] # Substitua com valores reais

train_errors = []
test_errors = []

epochs = 1000 # Aumentar o número de épocas
predictions_during_training = []
epochs.times do |epoch|
  epoch_train_error = 0
  x_normalized.each_with_index do |input, index|
    epoch_train_error += nn.train(input, y_normalized[index]).to_a.flatten.sum
  end
  train_errors << epoch_train_error / x_normalized.size
  
  predictions_normalized = test_inputs_normalized.map { |input| nn.predict(input) }
  predictions = denormalize(predictions_normalized.flatten, test_data.min, test_data.max)
  predictions_during_training << predictions
  test_error = mean_squared_error(real_values, predictions)
  test_errors << test_error

  puts "Epoch #{epoch + 1}/#{epochs} completo" if (epoch + 1) % 100 == 0
end

# Plotar os erros de treinamento e teste
Gnuplot.open do |gp|
  Gnuplot::Plot.new(gp) do |plot|
    plot.title "Erro de Treinamento e Teste ao Longo das Épocas"
    plot.xlabel "Épocas"
    plot.ylabel "Erro"
    plot.data << Gnuplot::DataSet.new(train_errors) do |ds|
      ds.with = "lines"
      ds.title = "Erro de Treinamento"
    end
    plot.data << Gnuplot::DataSet.new(test_errors) do |ds|
      ds.with = "lines"
      ds.title = "Erro de Teste"
    end
  end
end

# Plotar previsões vs valores reais
Gnuplot.open do |gp|
  Gnuplot::Plot.new(gp) do |plot|
    plot.title "Previsões vs Valores Reais"
    plot.xlabel "Mês"
    plot.ylabel "Precipitação"
    plot.data << Gnuplot::DataSet.new(real_values) do |ds|
      ds.with = "linespoints"
      ds.title = "Valores Reais"
    end
    plot.data << Gnuplot::DataSet.new(predictions) do |ds|
      ds.with = "linespoints"
      ds.title = "Previsões"
    end
  end
end

# Fazer previsões finais
predictions_normalized = test_inputs_normalized.map { |input| nn.predict(input) }
predictions = denormalize(predictions_normalized.flatten, test_data.min, test_data.max)
puts "Previsões: #{predictions}"

mae = mean_absolute_error(real_values, predictions)
mse = mean_squared_error(real_values, predictions)

puts "MAE: #{mae}"
puts "MSE: #{mse}"

# Plotar previsões finais vs valores reais
Gnuplot.open do |gp|
  Gnuplot::Plot.new(gp) do |plot|
    plot.title "Previsões Finais vs Valores Reais"
    plot.xlabel "Mês"
    plot.ylabel "Precipitação"
    plot.data << Gnuplot::DataSet.new(real_values) do |ds|
      ds.with = "linespoints"
      ds.title = "Valores Reais"
    end
    plot.data << Gnuplot::DataSet.new(predictions) do |ds|
      ds.with = "linespoints"
      ds.title = "Previsões Finais"
    end
  end
end

