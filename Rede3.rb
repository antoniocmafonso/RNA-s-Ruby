''' RNA - DNN '''

# Requerir as bibliotecas necessárias
require 'daru'
require 'numo/narray'
require 'dnn'
require 'dnn/image'
require 'dnn/layers'
require 'dnn/optimizers'
require 'dnn/callbacks'
require 'roo'
require 'gruff'
require 'numo/linalg'

# Carregar os dados do Excel
data = Roo::Spreadsheet.open('Varzea.xlsx')
df1 = Daru::DataFrame.new(data.parse(headers: true))

# Transformação dos dados
def min_max_scaler(data)
  min = data.min
  max = data.max
  (data - min) / (max - min)
end

scaled_data = df1.map_vectors { |vector| min_max_scaler(Numo::DFloat.cast(vector.to_a)) }

# Função para converter dados para matriz
def convert2matrix(data_arr, look_back)
  x, y = [], []
  (0...(data_arr.size - look_back)).each do |i|
    d = i + look_back
    x << data_arr[i...d].map(&:first)
    y << data_arr[d].first
  end
  [x, y]
end

# Funções de métricas
def r2_score(y_true, y_pred)
  ss_res = Numo::NArray.cast(y_true).zip(y_pred).map { |y_t, y_p| (y_t - y_p) ** 2 }.sum
  mean_y = y_true.sum / y_true.size
  ss_tot = y_true.map { |y| (y - mean_y) ** 2 }.sum
  1 - (ss_res / ss_tot)
end

def mean_squared_error(y_true, y_pred)
  Numo::NArray.cast(y_true).zip(y_pred).map { |y_t, y_p| (y_t - y_p) ** 2 }.sum / y_true.size
end

# Divisão dos dados em treino e teste
train_size = 487
train = scaled_data[0...train_size]
test = scaled_data[train_size..-1]

look_back = 24
train_x, train_y = convert2matrix(train, look_back)
test_x, test_y = convert2matrix(test, look_back)

# Definição do modelo
class DNNModel < DNN::Model
  def initialize(look_back)
    super()
    @dnn = DNN::Sequential.new
    @dnn << DNN::Layers::Dense.new(6, input_dim: look_back, activation: 'sigmoid')
    @dnn << DNN::Layers::Dense.new(6, activation: 'sigmoid')
    @dnn << DNN::Layers::Dense.new(1, activation: 'linear')
  end

  def forward(x)
    @dnn.call(x)
  end
end

model = DNNModel.new(look_back)
model.setup(DNN::Optimizers::Adam.new(learning_rate: 1e-3), loss: DNN::Losses::MeanSquaredError.new, metrics: ['mae'])

# Treinamento do modelo
callbacks = [DNN::Callbacks::EarlyStopping.new(monitor: :val_loss, patience: 5)]
history = model.fit(train_x, train_y, epochs: 1000, batch_size: 30, validation_data: [test_x, test_y], callbacks: callbacks, shuffle: false)

# Visualização da perda do modelo
loss_graph = Gruff::Line.new('800x600')
loss_graph.title = 'Model Loss'
loss_graph.data :Train_Loss, history.history[:loss]
loss_graph.data :Test_Loss, history.history[:val_loss]
loss_graph.labels = (0...history.history[:loss].length).to_a.each_with_index.map { |v, i| [i, v.to_s] }.to_h
loss_graph.write('loss.png')

# Previsões
train_predict = model.predict(train_x)
test_predict = model.predict(test_x)

# Visualização das previsões
train_graph = Gruff::Line.new('800x600')
train_graph.title = 'Previsão de Treinamento'
train_graph.data :Medido, train_y
train_graph.data :RNA, train_predict.to_a.map(&:first)
train_graph.labels = (0...train_y.length).to_a.each_with_index.map { |v, i| [i, v.to_s] }.to_h
train_graph.write('train_prediction.png')

# Métricas de treino
r2_train = r2_score(train_y, train_predict.to_a.map(&:first))
mse_train = mean_squared_error(train_y, train_predict.to_a.map(&:first))
puts "R² da RNA (treino): #{r2_train}"
puts "MSE da RNA (treino): #{mse_train}"

# Visualização das previsões de teste
test_graph = Gruff::Line.new('800x600')
test_graph.title = 'Previsão de Teste'
test_graph.data :Medido, test_y
test_graph.data :RNA, test_predict.to_a.map(&:first)
test_graph.labels = (0...test_y.length).to_a.each_with_index.map { |v, i| [i, v.to_s] }.to_h
test_graph.write('test_prediction.png')

# Métricas de teste
r2_test = r2_score(test_y, test_predict.to_a.map(&:first))
mse_test = mean_squared_error(test_y, test_predict.to_a.map(&:first))
puts "R² da RNA (teste): #{r2_test}"
puts "MSE da RNA (teste): #{mse_test}"

