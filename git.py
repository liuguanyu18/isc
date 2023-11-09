from datetime import datetime, timedelta
import warnings
import os

from pyspark import SparkConf
import logging
from pyspark.sql import SparkSession
import ast
import pandas as pd
import numpy as np
import time
from scipy import interpolate
import scipy.io as sio
# import datetime
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.ERROR)

os.environ["PYTHONPATH"] = "/home/hadoop/spark-3.2.1-bin-hadoop3.2/python/lib/py4j-0.10.6-src.zip"

os.environ['PYSPARK_PYTHON'] = './PY3/test_env/bin/python3'
os.environ['PYSPARK_DRIVER_PYTHON'] = './PY3/test_env/bin/python3'
os.environ['PYTHON_FAULTHANDLER_DIR'] = '/tmp/faulthandler'
os.environ["HADOOP_USER_NAME"] = "hive"
conf = SparkConf()
SPARK_EXECUTOR_MEMORY = '4g'
SPARK_EXECUTOR_CORES = 8
SPARK_EXECUTOR_INSTANCES = 40

env = [('spark.app.name', 'eddie_test'),
       ('spark.master', 'yarn'),
       ('spark.submit.deploymode', 'cluster'),
       ("spark.executor.memory", SPARK_EXECUTOR_MEMORY),
       ("spark.executor.cores", SPARK_EXECUTOR_CORES),
       ("spark.executor.instances", SPARK_EXECUTOR_INSTANCES),
       ('spark.driver.userClassPathFirst', 'true'),
       ('spark.files.ignoreMissingFiles', 'true'),
       ('spark.yarn.dist.archives', 'hdfs://hdfsHA/tmp/sd3/test_env.zip#PY3'),
       ('spark.driver.maxResultSize', 0),
       ('spark.shuffle.useOldFetchProtocol', 'true'),
       ('spark.dynamicAllocation.enabled', 'true'),
       ('spark.adaptive.repartition.enabled', 'true'),
       ('spark.yarn.queue', 'root.battery'),
       ('spark.python.worker.faulthandler.enabled', 'true'),
       ('spark.driver.memory', '8g')]

conf.setAll(env)

spark = SparkSession.builder.config(conf=conf).enableHiveSupport().getOrCreate()
spark.sparkContext.setLogLevel("ERROR")
class DataInput:
    def __init__(self):
        self.Temperature = None
        self.Temperature1 = None
        self.Voltage = None
        self.Voltage1 = None
        self.Time = None
        self.Current = None
        self.CAP = 150
        self.T_ambient = None
        self.VIN = None
        self.Volt_MIN = None
        self.Volt_AVG = None
        self.Temp_MAX = None
        self.Temp_AVG = None
        self.VMIN_ID = 1
        self.TMAX_ID = 1
        self.dt = None

    def string_to_float_list(self, data_input):
        # 如果输入是字符串，去掉括号并分割成列表
        if isinstance(data_input, str):
            try:
                float_values = [float(x.strip()) for x in data_input.strip("[]").split(",") if x.strip()]
                return float_values
            except ValueError as e:
                print(f"将字符串转换为浮点数列表时出错: {e}")
                return None
        elif isinstance(data_input, list):
            try:
                return [float(x) for x in data_input]
            except ValueError as e:
                print(f"将列表元素转换为浮点数时出错: {e}")
                return None
        else:
            print(f"不支持的输入类型: {type(data_input)}")
            return None
    def dataprocess(self, maxVoltage, minVoltage, cellVoltages, maxTemperature, minTemperature, cellTemperatures,
                    Current, Time,dt):
        #         print(f"Current value before conversion: {Current}, type: {type(Current)}")
        # 如果maxVoltage和minVoltage是字符串，先将它们转换成列表
        if isinstance(maxVoltage, str):
            maxVoltage = self.string_to_float_list(maxVoltage)
        if isinstance(minVoltage, str):
            minVoltage = self.string_to_float_list(minVoltage)

        # 然后取列表中的最大值和最小值作为最大电压和最小电压
        maxVoltage = max(maxVoltage) if isinstance(maxVoltage, list) else float(maxVoltage)
        minVoltage = min(minVoltage) if isinstance(minVoltage, list) else float(minVoltage)
        # 对maxTemperature和minTemperature执行相同的操作
        if isinstance(maxTemperature, str):
            maxTemperature = self.string_to_float_list(maxTemperature)
        if isinstance(minTemperature, str):
            minTemperature = self.string_to_float_list(minTemperature)

        # 然后取列表中的最大值和最小值作为最大温度和最小温度
        maxTemperature = max(maxTemperature) if isinstance(maxTemperature, list) else float(maxTemperature)
        minTemperature = min(minTemperature) if isinstance(minTemperature, list) else float(minTemperature)

        # 添加检查以确保cellVoltages和cellTemperatures不为None
        if cellVoltages is not None and isinstance(cellVoltages, str):
            cellVoltages = self.string_to_float_list(cellVoltages)
        if cellTemperatures is not None and isinstance(cellTemperatures, str):
            cellTemperatures = self.string_to_float_list(cellTemperatures)
        try:
            if isinstance(Current, str):  # 如果 Current 是字符串，尝试转换为浮点数
                Current = float(Current)
            elif not isinstance(Current, (int, float)):  # 如果 Current 不是数字或浮点数，打印错误并返回
                raise ValueError(f"Current is not a number: {Current}")
        except ValueError as e:
            print(f"Converting Current to float failed: {e}")
            return None

        Current = Current * (-1.0)
        Filed = {'Volt_MIN': np.nan, 'VMIN_ID': np.nan, 'Volt_AVG': np.nan, 'Temp_MAX': np.nan, 'TMAX_ID': np.nan,
                 'Temp_AVG': np.nan, 'Current': np.nan, 'T_ambient': np.nan, 'Time': np.nan,'dt':dt}
        # 添加检查以确保cellVoltages和cellTemperatures不为None
        cellVoltages_length = 0 if cellVoltages is None else len(cellVoltages)
        cellTemperatures_length = 0 if cellTemperatures is None else len(cellTemperatures)

        if cellVoltages_length == 0 or cellTemperatures_length == 0:
            self.method1(maxVoltage, minVoltage, maxTemperature, minTemperature, Current, Time,dt)
            print(self.Volt_MIN)
            Values = [self.Volt_MIN, self.VMIN_ID, self.Volt_AVG, self.Temp_MAX, self.TMAX_ID, self.Temp_AVG,
                      self.Current, self.T_ambient, self.Time,self.dt]
            for k, value in zip(Filed.keys(), Values):
                Filed[k] = value
            return Filed

        else:
            self.method2(cellVoltages, cellTemperatures, Current, Time, maxVoltage, minVoltage, maxTemperature,
                         minTemperature,dt)
            Values = [self.Volt_MIN, self.VMIN_ID, self.Volt_AVG, self.Temp_MAX, self.TMAX_ID, self.Temp_AVG,
                      self.Current, self.T_ambient, self.Time,self.dt]
            for k, value in zip(Filed.keys(), Values):
                Filed[k] = value
            return Filed

    def method1(self, avgVoltage, minVoltage, maxTemperature, minTemperature, Current, Time,dt):
        self.Volt_MIN = avgVoltage - 2 * (avgVoltage - minVoltage)
        self.Volt_AVG = minVoltage
        self.Temp_MAX = maxTemperature
        self.Temp_AVG = (maxTemperature + minTemperature) / 2
        self.T_ambient = minTemperature
        self.Current = Current
        self.dt = dt

        #         absTime1 = Time.strip(',').split(",")
        # 根据 Time 的类型进行处理
        if isinstance(Time, str):
            try:
                absTime1 = Time.strip(',').split(",")
                timearray = [self.parse_time(i) for i in absTime1]
            except Exception as e:
                print(f"Error while processing Time as string: {e}")
                self.Time = None  # 或其他默认值
                return

        elif isinstance(Time, (pd.Timestamp, datetime)):
            try:
                timearray = [self.parse_time(Time)]
            except Exception as e:
                print(f"Error while processing Time as Timestamp/datetime: {e}")
                self.Time = None  # 或其他默认值
                return

        else:
            print(f"Unsupported Time type: {type(Time)}")
            self.Time = None  # 或其他默认值
            return

        abstime_list = [int(time.mktime(t)) for t in timearray]
        self.Time = np.array(abstime_list)

    def method2(self, cellVoltages, cellTemperatures, Current, Time, maxVoltage, minVoltage, maxTemperature, minTemperature,dt):
        # 使用新的转换函数处理 cellVoltages 和 cellTemperatures
        self.Voltage1 = self.string_to_float_list(cellVoltages)
        self.Temperature1 = self.string_to_float_list(cellTemperatures)

        # 检查转换后的结果是否有效
        if not self.Voltage1 or not self.Temperature1:
            print("转换后没有有效的电压或温度数据。")
            return

        # 使用已清洗的电压值进行处理
        self.Voltage1 = [v if v > 3.35 else max(self.Voltage1) for v in self.Voltage1]

        # 解析时间
        Time = self.parse_time(Time)
        self.Time = int(time.mktime(Time))

        # 读取车辆ID
        self.VIN = 1
        self.dt = dt

        # 以下的计算中使用 self.Voltage1 和 self.Temperature1
        self.Voltage = list(np.array(self.Voltage1))
        self.Temperature = list(np.array(self.Temperature1))
        self.T_ambient = min(self.Temperature)  # 计算环境温度序列
        self.Current = 1.0 * Current  # 计算电流序列
        self.Volt_MIN = pd.Series(self.Voltage).min()
        self.VMIN_ID = pd.Series(self.Voltage).idxmin()
        self.Volt_AVG = (np.array(self.Voltage).sum() - self.Volt_MIN) / (len(np.array(self.Voltage)) - 1)
        self.Temp_MAX = pd.Series(self.Temperature1).max()
        self.TMAX_ID = pd.Series(self.Temperature1).idxmax()
        self.Temp_AVG = (np.array(self.Temperature1).sum() - self.Temp_MAX) / (len(np.array(self.Temperature1)) - 1)
        if (self.Volt_MIN == 0) & (self.Volt_AVG == 0):
            self.Volt_MIN = minVoltage
            self.Volt_AVG = (maxVoltage + minVoltage) / 2
        if (self.Temp_AVG == 0) & (self.Temp_MAX == 0):
            self.Temp_AVG = (maxTemperature + minTemperature) / 2
            self.Temp_MAX = maxTemperature

    def parse_time(self, timestr):
        if isinstance(timestr, str):
            try:
                return time.strptime(timestr, '%Y-%m-%d %H:%M:%S.%f')
            except ValueError:
                return time.strptime(timestr, '%Y-%m-%d %H:%M:%S')
        elif isinstance(timestr, (pd.Timestamp, datetime)):
            return timestr.timetuple()
        else:
            print(f"不支持的时间类型: {type(timestr)}")
            return None
def interpolate_fcn(x_break, y_break, table_data):
    values = np.transpose(table_data)
    z = interpolate.interp2d(x_break, y_break, values, kind='linear')
    return z


def interpolate_fcn_1d(x, values):
    x = x.reshape(-1)
    y = interpolate.interp1d(x, values, kind='linear', bounds_error=None, fill_value='extrapolate')
    return y
"""
InterpData为查表值字典
data为输入字典
EKF_P为delay的中间状态
"""


class ExtendedKalmanFilter:
    def __init__(self, EKF_P, data, InterpData):
        self.SOC_up_delay = EKF_P['SOC_up_delay']
        self.Cap = data['Cap']
        # self.time = EKF_P['Time']
        self.Current = data['Current']

        self.Current_Sign = 0
        self.abs_CRate = 0.0
        self.counter = EKF_P['counter']
        self.Lk_delay = EKF_P['Lk_delay']
        self.time_delay = EKF_P['time_delay']
        self.Tao_delay = EKF_P['Tao_delay']
        self.U1_delay = EKF_P['U1_delay']
        self.R1_delay = EKF_P['R1_delay']
        self.time = data['time']
        self.T_AVG = data['T_AVG']
        self.V_AVG = data['V_AVG']
        self.SGM_w = data['SGM_w']
        self.SGM_v = data['SGM_v']
        self.series = data['series']

        self.deltat = 1.0
        self.R0 = 0.0
        self.R1 = 0.0
        self.Tao = 0.0
        self.OCV = 0.0
        self.dVdS = 0.0
        self.alpha = 1.0
        self.Current_delay = EKF_P['Current_delay']
        # 需要打印出来.mat数据和hive表中数据核对 self.
        self.T_break = InterpData['T_break'].reshape(-1)
        self.CRate_break = InterpData['CRate_break'].reshape(-1)
        self.SOC_break = InterpData['SOC_break'].reshape(-1)
        self.TI2R0_C = InterpData['TI2R0_C']
        self.TI2R1_C = InterpData['TI2R1_C']
        self.TSOC2R0_C = InterpData['TSOC2R0_C']
        self.TSOC2R1_C = InterpData['TSOC2R1_C']
        self.TSOC2tao_C = np.fliplr(InterpData['TSOC2tao_C'])
        self.TI2R0_D = InterpData['TI2R0_D']
        self.TSOC2R0_D = InterpData['TSOC2R0_D']
        self.TI2R1_D = InterpData['TI2R1_D']
        self.TSOC2R1_D = InterpData['TSOC2R1_D']
        self.TSOC2tao_D = np.fliplr(InterpData['TSOC2tao_D'])
        self.TSOC2OCV = InterpData['TSOC2OCV']
        self.OCV_break = InterpData['OCV_break']
        self.TSOC2dVdS = InterpData['TSOC2dVdS']

    def delta_t(self):
        """
        实时计算数据的时间间隔
        """
        self.deltat = self.time - self.time_delay
        if (self.deltat > 100) | (self.deltat <= 0):
            self.deltat = 20
        return self.deltat

    def current2signate(self):
        """实时计算电流的方向及电池的倍率,充正放负"""
        if self.Current > 0:
            self.Current_Sign = 1
        else:
            self.Current_Sign = -1
        self.abs_CRate = np.abs(self.Current / self.Cap)

    def lookup_table(self):
        """计算插值函数，并在插值函数中实时计算插值点的函数值"""
        T_R0_C = interpolate_fcn(x_break=self.T_break, y_break=self.CRate_break, table_data=self.TI2R0_C)
        T_R1_C = interpolate_fcn(self.T_break, self.CRate_break, self.TI2R1_C)
        SOC_R0_C = interpolate_fcn(self.T_break, self.SOC_break, self.TSOC2R0_C)
        SOC_R1_C = interpolate_fcn(self.T_break, self.SOC_break, self.TSOC2R1_C)
        T_SOC2Tao_C = interpolate_fcn(self.T_break, self.SOC_break, self.TSOC2tao_C)
        T_R0_D = interpolate_fcn(self.T_break, self.CRate_break, self.TI2R0_D)
        SOC_R0_D = interpolate_fcn(self.T_break, self.SOC_break, self.TSOC2R0_D)
        T_R1_D = interpolate_fcn(self.T_break, self.CRate_break, self.TI2R1_D)
        SOC_R1_D = interpolate_fcn(self.T_break, self.SOC_break, self.TSOC2R1_D)
        T_SOC2Tao_D = interpolate_fcn(self.T_break, self.SOC_break, self.TSOC2tao_D)
        T_SOC2OCV = interpolate_fcn(self.T_break, self.SOC_break, self.TSOC2OCV)
        T_SOC2dVdS = interpolate_fcn(self.T_break, self.SOC_break, self.TSOC2dVdS)
        """根据电流方向，用不同的插值函数计算相应的函数值"""
        if self.Current_Sign >= 0:
            self.R0 = T_R0_C(self.T_AVG, self.abs_CRate)[0] * SOC_R0_C(self.T_AVG, self.SOC_up_delay)[0]
            self.R1 = T_R1_C(self.T_AVG, self.abs_CRate)[0] * SOC_R1_C(self.T_AVG, self.SOC_up_delay)[0]
            self.Tao = T_SOC2Tao_C(self.T_AVG, self.SOC_up_delay)[0]
        else:
            self.R0 = T_R0_D(self.T_AVG, self.abs_CRate)[0] * SOC_R0_D(self.T_AVG, self.SOC_up_delay)[0]
            self.R1 = T_R1_D(self.T_AVG, self.abs_CRate)[0] * SOC_R1_D(self.T_AVG, self.SOC_up_delay)[0]
            self.Tao = T_SOC2Tao_D(self.T_AVG, self.SOC_up_delay)[0]
        self.OCV = T_SOC2OCV(self.T_AVG, self.SOC_up_delay)[0]
        self.dVdS = T_SOC2dVdS(self.T_AVG, self.SOC_up_delay)[0]

    def ekf_cal(self):
        if self.V_AVG < min(self.OCV_break.tolist()[0]):
            self.V_AVG = min(self.OCV_break.tolist()[0])
        elif self.V_AVG > max(self.OCV_break.tolist()[0]):
            self.V_AVG = max(self.OCV_break.tolist()[0])
        """通过EKF估算SOC"""
        result = {'SOC_up': float, 'V_mdl': float}
        time1 = self.time - self.time_delay
        self.time_delay = self.time
        Flag = (self.counter <= 1) | (time1 > 1800)
        self.counter = 1 + self.counter
        """初始更新kalman滤波中的相关参数"""
        if self.Tao_delay != 0:
            self.alpha = np.exp(-self.deltat / self.Tao_delay)
        else:
            self.alpha = self.alpha
            self.Tao_delay = self.Tao

        Lk = (self.SGM_w + self.Lk_delay) * self.dVdS / (
                self.dVdS ** 2 * (self.SGM_w + self.Lk_delay) + self.SGM_v) / self.deltat
        self.Lk_delay = (1 - Lk * self.dVdS) * (self.SGM_w + self.Lk_delay)

        U1 = self.alpha * self.U1_delay + self.R1_delay * self.Current_delay * (1 - self.alpha)
        self.U1_delay = U1
        self.R1_delay = self.R1 / self.series
        # self.Current_delay = self.Current
        V_mdl = U1 + self.OCV + self.R0 * self.Current
        Ah = self.SOC_up_delay + self.deltat * self.Current_delay / self.Cap * 0.999 / 3600
        ERR = self.V_AVG - V_mdl
        SOC_up = Ah + Lk * ERR
        self.Current_delay = self.Current
        if SOC_up < 0:
            SOC_up = 0
        elif SOC_up > 1:
            SOC_up = 1

        if Flag:
            """当此orderID车辆初始化程序或时间间隔大于10800s时SOC值直接查表"""
            OCV2SOC = interpolate_fcn_1d(x=self.OCV_break.reshape(-1), values=self.SOC_break.reshape(-1))
            SOC_up = OCV2SOC(np.array([self.V_AVG]))[0]
            self.SOC_up_delay = 0.995 * self.SOC_up_delay + 0.005 * SOC_up
            result['SOC_up'] = SOC_up
            result['V_mdl'] = V_mdl
        else:
            if SOC_up > 1.0:
                SOC_up = 1.0
            self.SOC_up_delay = SOC_up
            result['SOC_up'] = SOC_up
            result['V_mdl'] = V_mdl
        return result

    def cal(self):

        if self.V_AVG < min(self.OCV_break.tolist()[0]):
            self.V_AVG = min(self.OCV_break.tolist()[0])
        elif self.V_AVG > max(self.OCV_break.tolist()[0]):
            self.V_AVG = max(self.OCV_break.tolist()[0])
        OCV2SOC = interpolate_fcn_1d(x=self.OCV_break.reshape(-1), values=self.SOC_break.reshape(-1))
        SOC_up = OCV2SOC(np.array([self.V_AVG]))[0]
        self.SOC_up_delay = SOC_up

    def main_cal(self):
        self.delta_t()
        self.current2signate()
        self.lookup_table()
        self.cal()
class LambdaFilter:
    """
    滤波对数据进行滤波
    """

    def __init__(self):
        self.result = 0.0

    def cal(self, Lambda, delay, data):
        self.result = Lambda * delay + (1 - Lambda) * data


# 计算产热功率
class PCalculation:
    """
    P_delay 为redis里存储的中间状态
    inputdata为实时输入的数据及参数
    其中：dt：时间间隔
    TMax：温度可以为最大值或平均值
    TAmb：环境温度
    Mass：电芯质量
    Cp：电池恒压比热容
    H：对流换热系数
    As：电芯热交换面积
    Lambda：递归最小二乘法的衰减系数
    LFDelay1：中间状态
    LFDelay2：中间状态
    """

    def __init__(self, P_delay, inputdata):
        default_value=0
        self.time = inputdata['time']
        self.time_delay = P_delay.get('time_delay',default_value)
        self.TMax = inputdata['TMax']
        self.TAmb = inputdata['TAmb']
        self.Mass = inputdata['Mass']
        self.Cp = inputdata['Cp']
        self.H = inputdata['H']
        self.As = inputdata['As']
        self.Lambda = inputdata['Lambda']
        self.TMax_delay = P_delay['TMax_delay']
        self.LFDelay1 = P_delay['Filter_delay1']
        self.LFDelay2 = P_delay['Filter_delay2']
        self.counter = P_delay['counter']
        self.counter_delay = P_delay['counter_delay']
        """实例化滤波类"""
        self.A = LambdaFilter()
        self.B = LambdaFilter()
        # 计算得到的产热功率
        self.P = 0.0

    # 计算产热功率
    def cal_p(self):
        dt = self.time - self.time_delay
        if dt <= 0:
            dt = 20
        if abs(self.TMax - self.TMax_delay) < 0.1:
            self.counter = self.counter + 1
        else:
            self.counter_delay = self.counter
            self.counter = 1

        # self.time_delay = self.time
        input1 = (self.TMax - self.TMax_delay) / dt / self.counter_delay
        input2 = self.TMax - self.TAmb
        self.A.cal(Lambda=self.Lambda, delay=self.LFDelay1, data=input1)
        self.B.cal(Lambda=self.Lambda, delay=self.LFDelay2, data=input2)
        # 更新delay的值
        self.TMax_delay = self.TMax
        self.LFDelay1 = self.A.result
        self.LFDelay2 = self.B.result
        # 计算产热功率
        self.P = self.A.result * self.Mass * self.Cp + self.B.result * self.H * self.As
# import numpy as np

"""建立报警的相关类"""


class WarningPSOC:
    """ 四级报警，可用于欧姆差及SOC差的报警状态计算
     th_1:报警升为一级的阈值
     th_2:报警升为二级的阈值
     th_3:报警升为三级的阈值
     th_4:报警升为四级的阈值
     th_down_1:由debounce状态降为0级报警的阈值
     th_down_2:由debounce状态降为1级报警的阈值
     th_down_3:由debounce状态降为2级报警的阈值
     th_down_4:由debounce状态降为3级报警的阈值
     time_th:报警升级时的debounce次数
     Timeth:表示报警debounce阈值
     Flag:判断是否是首次调用报警
     choice：表示报警状态，即在哪个函数中运行
     down1：表示由1级报警降到0级的阈值
     down2：表示由2级报警降到1级的阈值
     down3：表示由3级报警降到2级的阈值
     down4：表示由4级报警降到3级的阈值
     """
    def __init__(self, wpsoc_para, para):
        self.ROhm_Warn = 0
        self.time = int(para['Time_th'])
        self.Flag = int(wpsoc_para['Flag'])
        self.Delta_ROhm = 0
        self.switch = {}
        self.choice = wpsoc_para['choice']
        self.th_1 = para['th_1']
        self.th_2 = para['th_2']
        self.th_3 = para['th_3']
        self.th_4 = para['th_4']
        self.th_down_1 = para['th_down_1']
        self.th_down_2 = para['th_down_2']
        self.th_down_3 = para['th_down_3']
        self.th_down_4 = para['th_down_4']
        self.time_th = wpsoc_para['time_th']
        self.down1 = para['down1']
        self.down2 = para['down2']
        self.down3 = para['down3']
        self.down4 = para['down4']

    def in_normal(self):
        self.ROhm_Warn = 0
        if self.Delta_ROhm >= self.th_4:
            self.choice = 'Temp4'
            self.time = 0
        elif self.Delta_ROhm >= self.th_3:
            self.choice = 'Temp3'
            self.time = 0
        elif self.Delta_ROhm >= self.th_2:
            self.choice = 'Temp2'
            self.time = 0
        elif self.Delta_ROhm >= self.th_1:
            self.choice = 'Temp1'
            self.time = 0

    def in_temp1(self):
        if self.Delta_ROhm < self.th_down_1:
            self.choice = 'Normal'
            self.ROhm_Warn = 0
        elif (self.Delta_ROhm >= self.th_1) & (self.time >= self.time_th):
            self.choice = 'WARN1'
            self.ROhm_Warn = 1
        else:
            self.time += 1

    def in_temp2(self):
        if self.Delta_ROhm < self.th_down_2:
            self.choice = 'WARN1'
            self.ROhm_Warn = 1
        elif (self.Delta_ROhm >= self.th_2) & (self.time >= self.time_th):
            self.choice = 'WARN2'
            self.ROhm_Warn = 2
        else:
            self.time += 1

    def in_temp3(self):
        if self.Delta_ROhm < self.th_down_3:
            self.choice = 'WARN2'
            self.ROhm_Warn = 2
        elif (self.Delta_ROhm >= self.th_3) & (self.time >= self.time_th):
            self.choice = 'WARN3'
            self.ROhm_Warn = 3
        else:
            self.time += 1

    def in_temp4(self):
        if self.Delta_ROhm < self.th_down_4:
            self.choice = 'WARN3'
            self.ROhm_Warn = 3
        elif (self.Delta_ROhm >= self.th_4) & (self.time >= self.time_th):
            self.choice = 'WARN4'
            self.ROhm_Warn = 4
        else:
            self.time += 1

    def in_warn1(self):
        self.ROhm_Warn = 1
        if self.Delta_ROhm >= self.th_4:
            self.choice = 'Temp4'
            self.time = 0
        elif self.Delta_ROhm >= self.th_3:
            self.choice = 'Temp3'
            self.time = 0
        elif self.Delta_ROhm >= self.th_2:
            self.choice = 'Temp2'
            self.time = 0
        elif self.Delta_ROhm < self.down1:
            self.choice = 'Normal'

    def in_warn2(self):
        self.ROhm_Warn = 2
        if self.Delta_ROhm >= self.th_4:
            self.choice = 'Temp4'
            self.time = 0
        elif self.Delta_ROhm >= self.th_3:
            self.choice = 'Temp3'
            self.time = 0
        elif self.Delta_ROhm < self.down2:
            self.choice = 'WARN1'

    def in_warn3(self):
        self.ROhm_Warn = 3
        if self.Delta_ROhm >= self.th_4:
            self.choice = 'Temp4'
            self.time = 0
        elif self.Delta_ROhm < self.down3:
            self.choice = 'WARN2'

    def in_warn4(self):
        self.ROhm_Warn = 4
        if self.Delta_ROhm < self.down4:
            self.choice = 'WARN3'

    def rohm_warn(self, Delta_ROhm: float):
        self.Delta_ROhm = Delta_ROhm
        self.switch = {'Normal': self.in_normal,
                       'Temp1': self.in_temp1,
                       'Temp2': self.in_temp2,
                       'Temp3': self.in_temp3,
                       'Temp4': self.in_temp4,
                       'WARN1': self.in_warn1,
                       'WARN2': self.in_warn2,
                       'WARN3': self.in_warn3,
                       'WARN4': self.in_warn4
                       }

        if self.Flag == 0:
            self.Flag = 1
            self.ROhm_Warn = 0
            self.choice = 'Normal'
            return self.ROhm_Warn
        else:
            self.switch.get(self.choice)()
            return self.ROhm_Warn


class WaringVT:
    """ 压差及温差报警等级判断
    th_1:报警升为一级的阈值
    th_2:报警升为二级的阈值
    th_down_1:由debounce状态降为0级报警的阈值
    th_down_2:由debounce状态降为1级报警的阈值
    Delta_V_up: 报警等级升高时的阈值
    Delta_V_down： 报警等级下降时的阈值
    time_th:进入报警前的debounce次数
    down1：表示由1级报警降到0级的阈值
    down2：表示由2级报警降到1级的阈值
    time_th:报警升级时的debounce次数
    Timeth:表示报警debounce阈值
    Flag:判断是否是首次调用报警
    choice：表示报警状态，即在哪个函数中运行
    """
    def __init__(self, wvt_para, para):
        self.Warn = 0
        self.time = int(wvt_para['time_th'])
        self.Delta = 0
        self.Delta_up1 = para['th_1']
        self.Delta_down1 = para['th_down_1']
        self.Delta_up2 = para['th_2']
        self.Delta_down2 = para['th_down_2']
        self.Flag = int(wvt_para['Flag'])
        self.choice = wvt_para['choice']
        self.switch = {}
        self.time_th = para['Time_th']
        self.down1 = para['down1']
        self.down2 = para['down2']

    def normal(self):
        self.Warn = 0
        self.time = 0
        if self.Delta >= self.Delta_up2:
            self.choice = 'temp2'
        elif self.Delta >= self.Delta_up1:
            self.choice = 'temp1'

    def temp1(self):
        if self.Delta < self.Delta_down1:
            self.choice = 'normal'
            self.Warn = 0
        elif self.time >= self.time_th:
            self.choice = 'warn1'
            self.Warn = 1
        else:
            self.time += 1

    def temp2(self):
        if self.Delta < self.Delta_down2:
            self.choice = 'warn1'
            self.Warn = 1
        elif self.time >= self.time_th:
            self.choice = 'warn2'
            self.Warn = 2
        else:
            self.time += 1

    def warn1(self):
        self.time = 0
        if self.Delta < self.down1:
            self.choice = 'normal'
            self.Warn = 0
        elif self.Delta >= self.Delta_up2:
            self.choice = 'temp2'
            self.Warn = 1

    def warn2(self):
        self.Warn = 2
        if self.Delta < self.down2:
            self.choice = 'warn1'
            self.Warn = 1

    def entry(self, Delta_V):
        self.Delta = Delta_V
        self.switch = {'normal': self.normal,
                       'temp1': self.temp1,
                       'temp2': self.temp2,
                       'warn1': self.warn1,
                       'warn2': self.warn2}
        if self.Flag == 0:
            self.Flag = 1
            self.Warn = 0
            self.choice = 'normal'
            return self.Warn
        else:
            self.switch.get(self.choice)()
            return self.Warn


def warn_flag(WT, WP, WSOC, WV):
    if WT + WP + WSOC + WV >= 5:
        return 1
    else:
        return 0


def warn_cap(WT, WP):
    if WT + WP >= 2:
        return 1
    else:
        return 0
class MainSafe:
    def __init__(self, par, delay_data, inputdata, InterpData):
        self.Flag = 0
        self.Cap_Flag = 0
        self.Heat_Flag = 0
        self.deltaSoc = delay_data['deltaSoc']
        self.deltaV = delay_data['deltaV']

        self.T_max_delay = delay_data['delay_PMax']['T_max_delay']
        self.counter_f = delay_data['delay_PMax']['counter_f']
        self.T_max = inputdata['pMax_p']['TMax']
        inputdata['pMax_p']['TMax'] = self.filter_temperature()
        self.pMax = PCalculation(P_delay=delay_data['delay_PMax'], inputdata=inputdata['pMax_p'])
        self.pAve = PCalculation(P_delay=delay_data['delay_PAvg'], inputdata=inputdata['pAvg_p'])
        self.SOCAve = ExtendedKalmanFilter(EKF_P=delay_data['delay_SOCAvg'], data=inputdata['SOCAvg_p'],
                                           InterpData=InterpData)
        self.SOCMin = ExtendedKalmanFilter(EKF_P=delay_data['delay_SOCMin'], data=inputdata['SOCMin_p'],
                                           InterpData=InterpData)
        self.WP = WarningPSOC(delay_data['wp'], par['wp_p'])
        self.WSOC = WarningPSOC(delay_data['wsoc'], par['wsoc_p'])
        self.WV = WaringVT(delay_data['wv'], par['wv_p'])
        self.WT = WaringVT(delay_data['wt'], par['wt_p'])
        self.WTT = WarningPSOC(delay_data['wtt'], par['wtt_p'])
        self.Delta_V = inputdata['Delta_V']
        self.Delta_T = inputdata['Delta_T']
        self.orderid = inputdata['orderid']

    def set_parm(self):
        pass

    def filter_temperature(self):
        if self.counter_f == 0:
            self.T_max_delay = self.T_max
        if self.T_max_delay == self.T_max:
            self.counter_f = 1
        else:
            self.counter_f = self.counter_f + 1
        if self.counter_f > 10:
            self.T_max_delay = self.T_max
        return self.T_max_delay

    def calculation(self):
        self.pMax.cal_p()
        self.pAve.cal_p()
        self.SOCAve.main_cal()
        self.SOCMin.main_cal()
        self.WP.rohm_warn(self.pMax.P - self.pAve.P)
        self.deltaSoc = 0.99 * self.deltaSoc + (1 - 0.99) * (self.SOCAve.SOC_up_delay - self.SOCMin.SOC_up_delay)
        self.WSOC.rohm_warn(self.deltaSoc)
        self.deltaV = 0.9 * self.deltaV + (1 - 0.9) * self.Delta_V
        self.WTT.rohm_warn(self.deltaV)

        self.WV.entry(self.Delta_V)

        self.WT.entry(self.Delta_T)
        self.Flag = warn_flag(self.WTT.ROhm_Warn, self.WSOC.ROhm_Warn, self.WV.Warn, self.WT.Warn)
        self.Cap_Flag = warn_cap(self.WV.Warn, self.WSOC.ROhm_Warn)
        self.Heat_Flag = warn_cap(self.WT.Warn, self.WTT.ROhm_Warn)
        delaystate = self.save_state()
        return delaystate

    def save_state(self):
        delay_PMax = {'Filter_delay1': self.pMax.LFDelay1, 'Filter_delay2': self.pMax.LFDelay2,
                      'TMax_delay': self.pMax.TMax_delay, '_delay': self.pMax.time,
                      'counter': self.pMax.counter, 'counter_delay': self.pMax.counter_delay,
                      'counter_f': self.counter_f, 'T_max_delay': self.T_max_delay}
        delay_SOCAvg = {'SOC_up_delay': self.SOCAve.SOC_up_delay, 'counter': self.SOCAve.counter,
                        'Lk_delay': self.SOCAve.Lk_delay, 'time_delay': self.SOCAve.time_delay,
                        'Tao_delay': self.SOCAve.Tao_delay, 'U1_delay': self.SOCAve.U1_delay,
                        'R1_delay': self.SOCAve.R1_delay, 'Current_delay': self.SOCAve.Current_delay}
        delay_PAvg = {'Filter_delay1': self.pAve.LFDelay1, 'Filter_delay2': self.pAve.LFDelay2,
                      'TMax_delay': self.pAve.TMax_delay, 'time_delay': self.pAve.time,
                      'counter': self.pAve.counter, 'counter_delay': self.pAve.counter_delay}
        delay_SOCMin = {'SOC_up_delay': self.SOCMin.SOC_up_delay, 'counter': self.SOCMin.counter,
                        'Lk_delay': self.SOCMin.Lk_delay, 'time_delay': self.SOCMin.time_delay,
                        'Tao_delay': self.SOCMin.Tao_delay, 'U1_delay': self.SOCMin.U1_delay,
                        'R1_delay': self.SOCMin.R1_delay, 'Current_delay': self.SOCMin.Current_delay}
        wp = {'time_th': self.WP.time_th, 'Flag': self.WP.Flag, 'choice': self.WP.choice}
        wsoc = {'time_th': self.WSOC.time_th, 'Flag': self.WSOC.Flag, 'choice': self.WSOC.choice}
        wv = {'Flag': self.WV.Flag, 'time_th': self.WV.time_th, 'choice': self.WV.choice}
        wt = {'Flag': self.WT.Flag, 'time_th': self.WT.time_th, 'choice': self.WT.choice}
        wtt = {'time_th': self.WTT.time_th, 'Flag': self.WTT.Flag, 'choice': self.WTT.choice}
        delay_state = {'delay_PMax': delay_PMax, 'delay_SOCAvg': delay_SOCAvg, 'delay_PAvg': delay_PAvg,
                       'delay_SOCMin': delay_SOCMin, 'wp': wp, 'wsoc': wsoc, 'wv': wv, 'wt': wt, 'wtt': wtt,
                       'deltaSoc': self.deltaSoc, 'deltaV': self.deltaV}
        return delay_state

    def maincal(self):
        delay1 = self.calculation()
        return delay1


class SafeAlarm:
    def __init__(self, params=None):
        """delay的状态"""
        self.results = []  # 初始化一个空列表来存储所有结果

        delay_PMax = {'Filter_delay1': 0, 'Filter_delay2': 0, 'time_delay': 0, 'TMax_delay': 0.0, 'counter': 1,
                      'counter_delay': 1, 'counter_f': 0, 'T_max_delay': 0.0}
        delay_SOCAvg = {'SOC_up_delay': 0, 'counter': 0, 'Lk_delay': 0, 'time_delay': 0, 'Tao_delay': 1, 'U1_delay': 0,
                        'R1_delay': 0, 'Current_delay': 0}
        delay_PAvg = {'Filter_delay1': 0, 'Filter_delay2': 0, 'time_delay': 0, 'TMax_delay': 0.0, 'counter': 1,
                      'counter_delay': 1}
        delay_SOCMin = {'SOC_up_delay': 0, 'counter': 0, 'Lk_delay': 0, 'time_delay': 0, 'Tao_delay': 1, 'U1_delay': 0,
                        'R1_delay': 0, 'Current_delay': 0}
        wp = {'time_th': 0, 'Flag': 0, 'choice': 'normal'}
        wsoc = {'time_th': 0, 'Flag': 0, 'choice': 'normal'}
        wv = {'Flag': 0, 'time_th': 0, 'choice': 'normal'}
        wt = {'Flag': 0, 'time_th': 0, 'choice': 'normal'}
        wtt = {'time_th': 0, 'Flag': 0, 'choice': 'normal'}
        self.delay = {'delay_PMax': delay_PMax, 'delay_SOCAvg': delay_SOCAvg, 'delay_PAvg': delay_PAvg, 'wtt': wtt,
                      'delay_SOCMin': delay_SOCMin, 'wp': wp, 'wsoc': wsoc, 'wv': wv, 'wt': wt, 'deltaSoc': 0.0,
                      'deltaV': 0.0}
        """可配的参数"""
        wp_p = {'th_1': 0.53 + 0.5, 'th_2': 0.73 + 0.5, 'th_3': 0.93 + 0.5, 'th_4': 1.13 + 0.5, 'th_down_1': 0.43 + 0.5,
                'th_down_2': 0.63 + 0.5,
                'th_down_3': 0.83 + 0.5, 'th_down_4': 1.03 + 0.5, 'down1': 0.35 + 0.5, 'down2': 0.55 + 0.5,
                'down3': 0.75 + 0.5, 'down4': 0.95 + 0.5,
                'Time_th': 5}
        wsoc_p = {'th_1': 0.066, 'th_2': 0.106, 'th_3': 0.146, 'th_4': 0.186, 'th_down_1': 0.056,
                  'th_down_2': 0.096,
                  'th_down_3': 0.136, 'th_down_4': 0.176, 'down1': 0.046, 'down2': 0.086, 'down3': 0.126,
                  'down4': 0.166,
                  'Time_th': 5}
        wv_p = {'th_1': 0.07, 'th_down_1': 0.05, 'th_2': 0.12, 'th_down_2': 0.1, 'down1': 0.03, 'down2': 0.08,
                'Time_th': 5}
        wt_p = {'th_1': 8.0, 'th_down_1': 7.0, 'th_2': 12.0, 'th_down_2': 11.0, 'down1': 6.0, 'down2': 10.0,
                'Time_th': 5}
        wtt_p = {'th_1': 7.0, 'th_2': 11.0, 'th_3': 15.0, 'th_4': 19.0, 'th_down_1': 6.0, 'th_down_2': 10.0,
                 'th_down_3': 14.0, 'th_down_4': 18.0, 'down1': 5.0, 'down2': 9.0, 'down3': 13.0, 'down4': 17.0,
                 'Time_th': 3}
        cell_p = {'Mass': 1.0, 'Cp': 900.0, 'H': 15, 'As': 0.01, 'Lambda': 0.999, 'Cap': 150.0, 'SGM_w': 0.01,
                  'SGM_v': 0.001, 'series': 1}
        self.default_params = {'wp_p': wp_p, 'wsoc_p': wsoc_p, 'wv_p': wv_p, 'wt_p': wt_p, 'cell_p': cell_p,
                               'wtt_p': wtt_p,
                               'path': './Lookup_table.mat'
                               }
        self.setParams(params)

    def setParams(self, params):
        if params is None:
            params = {}
        for k, v in params.items():  # 注意这里使用.items()来正确解包params字典
            self.default_params[k] = v

    def safealarm(self, data, params=None) -> dict:
        self.setParams(params)
        result = self.method1(data)
        return result

    def safe_cal(self, dataline, delay, i):
        if dataline['cell_min_volt'] is None:
            return "SKIP", None  # 返回一个标识符SKIP来告知主调函数跳过这条数据
        # 在这里添加检查，确保没有None值被转换为float
        min_voltage = dataline['cell_min_volt']
        max_temp = dataline['cell_max_tem']

        # 如果min_voltage或max_temp是None，可以选择跳过或提供默认值
        if min_voltage is None or max_temp is None:
            return "SKIP", None  # 跳过这条记录
        data = DataInput()
        data.dataprocess(dataline['cell_max_volt'], dataline['cell_min_volt'],
                         dataline['bmv_singlevolt'],
                         dataline['cell_max_tem'],
                         dataline['cell_min_tem'], dataline['bmt_singletemp'],
                         dataline['bcs_currentmeasure'],
                         dataline['reltime'],
                         dataline['dt'])
        interp_data = sio.loadmat(self.default_params['path'])
        """数据输入"""
        # inputda = {'orderId', 'Delta_V', 'Delta_T', 'dt', 'TMax', 'TAmb', 'Mass', 'Cp', 'H', 'As', 'Lambda'}
        cell_p = self.default_params['cell_p']
        pMax_p = {'time': data.Time, 'TMax': data.Temp_MAX, 'TAmb': data.T_ambient, 'Mass': cell_p['Mass'],
                  'Cp': cell_p['Cp'], 'H': cell_p['H'], 'As': cell_p['As'], 'Lambda': cell_p['Lambda']}
        pAvg_p = {'time': data.Time, 'TMax': data.Temp_AVG, 'TAmb': data.T_ambient, 'Mass': cell_p['Mass'],
                  'Cp': cell_p['Cp'], 'H': cell_p['H'], 'As': cell_p['As'], 'Lambda': cell_p['Lambda']}
        SOCAvg_p = {'Cap': cell_p['Cap'], 'Current': data.Current, 'time': data.Time, 'T_AVG': data.Temp_AVG,
                    'V_AVG': data.Volt_AVG, 'SGM_w': cell_p['SGM_w'], 'SGM_v': cell_p['SGM_v'],
                    'series': cell_p['series']}
        SOCMin_p = {'Cap': cell_p['Cap'], 'Current': data.Current, 'time': data.Time, 'T_AVG': data.Temp_MAX,
                    'V_AVG': data.Volt_MIN, 'SGM_w': cell_p['SGM_w'], 'SGM_v': cell_p['SGM_v'],
                    'series': cell_p['series']}

        input_data = {'pMax_p': pMax_p, 'pAvg_p': pAvg_p, 'SOCAvg_p': SOCAvg_p, 'SOCMin_p': SOCMin_p,
                      'orderid': dataline['orderid'],
                      'Delta_V': data.Volt_AVG - data.Volt_MIN, 'Delta_T': data.Temp_MAX - data.Temp_AVG}
        try:
            A2 = MainSafe(par=self.default_params, delay_data=delay, inputdata=input_data, InterpData=interp_data)
        except TypeError as e:
            #             print("Error: Problem with delay_data. Please check its value.")
            # 如果你还希望看到原始错误消息，可以添加以下行：
            #             print("Original error:", e)
            return "ERROR",None  # 或 raise e
        delay = A2.maincal()
        result_end = {
            'time': dataline['reltime'],
            'orderid': dataline['orderid'],'batteryid':dataline['batteryid'], 'ISC_FAULT': A2.Flag,
            'Cap_dep_FAULT': A2.Cap_Flag, 'Heat_gen_FAULT': A2.Heat_Flag,
            'MinVoltage': float(data.Volt_MIN), 'TempId': data.TMAX_ID, 'MaxTemp': float(data.Temp_MAX),
            'DeltaP': float(A2.pMax.P - A2.pAve.P), 'TMax': A2.pMax.TMax_delay, 'TAve': A2.pAve.TMax_delay,'dt': dataline['dt']}
        return result_end, delay

    def method1(self, data=None):
        #         data=sdf.toPandas()
        print("Starting method1...")
        sdf = spark.createDataFrame(data)
        results = []
        ISC_FAULT = 0
        Cap_dep_FAULT = 0
        Heat_gen_FAULT = 0
        result1 = {'reltime': None, 'orderid': None, 'ISC_FAULT': None, 'Cap_dep_FAULT': None, 'Heat_gen_FAULT': None,
                   'batteryid': None, 'MinVoltage': None, 'TempId': None, 'MaxTemp': None, 'DeltaP': None,'dt': None}
        if sdf.shape[0] < 1:
            result = {'error': 599}
        else:
            result = {}
        for i in range(len(sdf)):
            da = sdf.iloc[i]

            result, self.delay = self.safe_cal(da, self.delay, i)
            # 检查是否要跳过这条数据
            if result == "SKIP":
                continue
            if result == "ERROR":
                print("Error encountered in safe_cal. Skipping this entry.")
                continue
            ISC_FAULT = result['ISC_FAULT'] + ISC_FAULT
            Cap_dep_FAULT = result['Cap_dep_FAULT'] + Cap_dep_FAULT
            Heat_gen_FAULT = result['Heat_gen_FAULT'] + Heat_gen_FAULT
            if (result['ISC_FAULT'] > 0) | (result['Cap_dep_FAULT'] > 0) | (result['Heat_gen_FAULT'] > 0):
                result1 = result
            if result1['orderid'] is None:
                result1 = result
            if sdf.shape[0] >= 1:
                result1['ISC_RATE'] = ISC_FAULT / sdf.shape[0]
                result1['Cap_dep_RATE'] = Cap_dep_FAULT / sdf.shape[0]
                result1['Heat_gen_RATE'] = Heat_gen_FAULT / sdf.shape[0]
                if result1['ISC_RATE'] > 0:
                    result1['ISC_FAULT_ret'] = 2
                else:
                    result1['ISC_FAULT_ret'] = 1
                if result1['Cap_dep_RATE'] > 0:
                    result1['Cap_dep_FAULT_ret'] = 2
                else:
                    result1['Cap_dep_FAULT_ret'] = 1
                if result1['Heat_gen_RATE'] > 0:
                    result1['Heat_gen_FAULT_ret'] = 2
                else:
                    result1['Heat_gen_FAULT_ret'] = 1
                result_dict = {
                    'reltime': result['time'],
                    'orderid': result1['orderid'],
                    'ISC_FAULT': result['ISC_FAULT'],
                    'Cap_dep_FAULT': result['Cap_dep_FAULT'],
                    'Heat_gen_FAULT': result['Heat_gen_FAULT'],
                    'batteryid': result['batteryid'],
                    'MinVoltage': result['MinVoltage'],
                    'TempId': result['TempId'],
                    'MaxTemp': result['MaxTemp'],
                    'DeltaP': result['DeltaP'],
                    'Heat_gen_FAULT_ret': result1['Heat_gen_FAULT_ret'],
                    'Cap_dep_FAULT_ret': result1['Cap_dep_FAULT_ret'],
                    'ISC_FAULT_ret': result1['ISC_FAULT_ret'],'dt': result1['dt']}
                results.append(result_dict)

        #         results_df = pd.DataFrame(results)
        results_df = spark.createDataFrame(results)
        return results_df
from pyspark.sql import SparkSession, functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType
import sys

def process_order(order_df):
    try:
        order_df = order_df.withColumn('cell_min_volt', F.col('cell_min_volt').cast(DoubleType()))
        order_df = order_df.withColumn('cell_max_tem', F.col('cell_max_tem').cast(DoubleType()))
        #         pandas_df = order_df.toPandas()

        alarm = SafeAlarm()
        order_result = alarm.safealarm(data=order_df)

        if not order_result.empty:
            result_spark_df = spark.createDataFrame(order_result)
            result_spark_df.write.mode('append').saveAsTable("battery_security.pfp_result_isc_event_details_ir_997_new_test")

        # 手动释放内存
        order_df.unpersist()

    except Exception as e:
        print(f"处理订单 {order_df.first()['orderid']} 时出错: {str(e)}")
        return False
    return True
#测试新思路 修改表名
def process_data(start_date, end_date):
    failed_orders=[]
    sdf = spark.sql(f"""
        SELECT *
        FROM battery_security.result_isc_event_details_ir_input_997
        WHERE dt BETWEEN '{start_date}' AND '{end_date}'
    """)

    if sdf.rdd.isEmpty():
        return [start_date, end_date]

    window_spec = Window.partitionBy("orderid").orderBy("reltime")
    sdf = sdf.withColumn("row_number", F.row_number().over(window_spec))
    order_ids = sdf.select("orderid").distinct().collect()

    # 处理两个 orderid 为一组
    for i in range(0, len(order_ids), 100):
        order_id_group = order_ids[i:i+100]  # 获取两个 orderid
        for order_id_row in order_id_group:
            order_id = order_id_row['orderid']
            order_df = sdf.filter(F.col("orderid") == order_id).orderBy("reltime")
            successful = process_order(order_df)
            if not successful:
                failed_orders.append(order_id)

    return failed_orders

# 主流程
# start_date = sys.argv[1]
# end_date = sys.argv[2]
start_date = '2023-08-04'
end_date = '2023-08-04'
failed_orders = process_data(start_date, end_date)
if failed_orders:
    print(f"处理失败的订单ID：{failed_orders}")
else:
    print("指定日期范围内的所有订单数据处理完成并已存入结果表。")
