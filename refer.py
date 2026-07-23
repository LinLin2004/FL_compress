import pandas as pd
import io
from openpyxl.styles import Font

# --- 完整的 CSV 数据 (包含简洁表头和42条文献数据) ---
csv_data = """
"Citation Key","Year","Venue/Source","Paper Category","Goal: Comm. Efficiency","Goal: Robustness/Byzantine Security","Goal: Acceleration/Convergence","Goal: Privacy/Incentive","Focus on Non-IID Data","Focus on System Heterogeneity","Tech: Local Updates","Tech: Compression/Quantization","Tech: Sparsification/Top-K","Tech: Robust Aggregation","Tech: Momentum/Nesterov","Tech: Adaptive Optimizer","Theoretical Convergence Guarantee","Application Domain","Full Title"
"mcmahan2023communication","2023","arXiv","Research (Foundational Re-release)","Yes","","Yes","","Yes (Empirically)","Yes","Yes","","","","","","Empirical Focus","General FL","Communication-Efficient Learning of Deep Networks from Decentralized Data"
"rieke2020the","2020","npj Digital Medicine","Survey/Position Paper","","","","Yes","Yes","","","","","","","","N/A (Survey)","Digital Health","The future of digital health with federated learning"
"PATI2024100974","2024","Patterns","Research","","","","Yes","Yes","","","","","","","","","Healthcare (Privacy Focus)","Privacy preservation for federated learning in health care"
"WANG2024114084","2024","Decision Support Systems","Research","","","","","Yes","","","","","","","","","Finance (Credit Scoring)","A novel federated learning approach with knowledge transfer for credit scoring"
"lim2020federated","2020","IEEE Commun. Surv. Tutorials","Survey","Yes","","","Yes","","Yes","","","","","","","N/A (Survey)","Mobile Edge Networks","Federated Learning in Mobile Edge Networks: A Comprehensive Survey"
"niknam2020federated","2020","IEEE Communications Magazine","Survey/Magazine","Yes","","","","","Yes","","","","","","","N/A (Survey)","Wireless Communications","Federated Learning for Wireless Communications: Motivation, Opportunities, and Challenges"
"pokhrel2021federated","2021","IEEE Internet Things J.","Survey","","","","","","Yes","","","","","","","N/A (Survey)","Edge Computing & Blockchain","Federated Learning Meets Blockchain in Edge Computing: Opportunities and Challenges"
"ye2020federated","2020","IEEE Network","Survey","","","","","","Yes","","","","","","","N/A (Survey)","Vehicular Edge Computing","Federated Learning in Vehicular Edge Computing: A Survey"
"blanchard2017machine","2017","NeurIPS","Research (Foundational)","","Yes","","","","","","","","Yes (Krum/Multi-Krum)","","","Yes","General Distributed ML","Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent"
"yin2018byzantine","2018","ICML","Research","","Yes","","","","","","","","Yes (Trimmed Mean/Median)","","","Yes","General Distributed ML","Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates"
"konecný2016federated","2016","CoRR (arXiv)","Research","Yes","","","","","","Yes","Yes (Structured/Sketched)","Yes","","","","Yes","General FL (On-Device)","Federated Optimization: Distributed Machine Learning for On-Device Intelligence"
"alistarh2017qsgd","2017","NeurIPS","Research","Yes","","","","","","","Yes (QSGD)","","","","","Yes","General Distributed ML","QSGD: Communication-Efficient SGD via Gradient Quantization and Encoding"
"stich2018sparsified","2018","NeurIPS","Research","Yes","","Yes (via Memory)","","","","","","Yes (with error correction)","","","","Yes","General Distributed ML","Sparsified SGD with Memory"
"allouah2025adaptive","2025","ICLR","Research","","Yes","","","","","","","","Yes (Adaptive Clipping)","","","Yes","General FL","Adaptive Gradient Clipping for Robust Federated Learning"
"chen2025allreducecompatibletopkcompressor","2025","arXiv","Research","Yes","","","","","","","","Yes","Yes (Top-K)","","","","Yes","General Distributed Learning","An All-Reduce Compatible Top-K Compressor for Communication-Efficient Distributed Learning"
"kingma2015adam","2015","ICLR","Research (Foundational)","","","Yes","","","","","","","","","","Yes (Adam Origin)","Yes","General Optimization","Adam: A Method for Stochastic Optimization"
"aji2017sparse","2017","ICLR Workshop","Research","Yes","","","","","","","","","Yes (Gradient Dropping)","","","","Empirical Focus","General Distributed GD","Sparse Communication for Distributed Gradient Descent"
"lin2018deep","2018","ICLR","Research","Yes","","","","","","","","","Yes (Deep Gradient Compression)","","","","Empirical Focus","General Distributed Training","Deep Gradient Compression: Reducing the Communication Bandwidth for Distributed Training"
"Fu2024","2024","IEEE Internet of Things Journal","Research","","","","Yes (Incentive)","","Yes","","","","","","","","Autonomous Driving","An Incentive Mechanism for Long-Term Federated Learning in Autonomous Driving"
"Lin2024","2024","IROS","Research","","","","","Yes","","","","","","","","","Autonomous Driving (Planning)","PP-TIL: Personalized Planning for Autonomous Driving with Instance-based Transfer Imitation Learning"
"xu2022healthcare","2022","Journal of Healthcare Informatics Research","Survey","","","","Yes","Yes","","","","","","","","N/A (Survey)","Healthcare Informatics","Federated Learning for Healthcare Informatics"
"Shangguan2025","2025","ACM Computing Surveys","Survey","","","","","","Yes","","","","","","","N/A (Survey)","IoT (Facial Expression)","Facial Expression Analysis and Its Potentials in IoT Systems: A Contemporary Survey"
"mhamdi2018bulyan","2018","ICML","Research","","Yes","","","","","","","","Yes (Bulyan Aggregation)","","","Yes","General Distributed ML","The Hidden Vulnerability of Distributed Learning in Byzantium"
"yang2020fednag","2022","IEEE Trans. on Parallel and Distributed Systems","Research","","","Yes","","","","","","","","","Yes (Nesterov)","","Yes","General FL","Federated Learning With Nesterov Accelerated Gradient"
"reddi2021adaptive","2021","ICLR","Research","","","Yes","","Yes (Primary Focus)","","","","","","","","Yes (FedOpt/FedAdam)","Yes","General FL","Adaptive Federated Optimization"
"Wu2023","2023","AAAI","Research","","","Yes","","Yes","","","","","","","","Yes (Faster Adaptive)","Yes","General FL","Faster adaptive federated learning"
"Yu2019","2019","ICML","Research","Yes","","Yes","","","","","","","","","Yes (Momentum)","","Yes","General Distributed Non-Convex","On the Linear Speedup Analysis of Communication Efficient Momentum SGD for Distributed Non-Convex Optimization"
"chen2017distributed","2017","ACM Measurement and Analysis","Research","","Yes","","","","","","","","Yes (Byzantine GD)","","","Yes","General Distributed ML","Distributed Statistical Machine Learning in Adversarial Settings: Byzantine Gradient Descent"
"Farhadkhani2022Resam","2022","ICML","Research","","Yes","","","","","","","","Yes","Yes (Resilient Averaging)","","Yes","General ML","Byzantine Machine Learning Made Easy By Resilient Averaging of Momentums"
"chen2018lag","2018","NeurIPS","Research","Yes","","","","","","","","","","","","Yes","General Distributed Learning","LAG: Lazily Aggregated Gradient for Communication-Efficient Distributed Learning"
"stich2019local","2019","ICLR","Research","Yes","","Yes","","","","Yes (Analysis of Local SGD)","","","","","","Yes","General Distributed Learning","Local SGD Converges Fast and Communicates Little"
"yu2019linear","2019","ICML","Research","Yes","","Yes","","","","","","","","","Yes (Momentum)","","Yes","General Distributed Non-Convex","On the Linear Speedup Analysis of Communication Efficient Momentum SGD for Distributed Non-Convex Optimization"
"mao2022communication","2022","ACM Trans. on Intelligent Systems and Technology","Research","Yes","","","","Yes","","","Yes (Adaptive Quantization)","","","","","Yes","General FL","Communication-Efficient Federated Learning with Adaptive Quantization"
"zheng2024communication","2024","Applied Intelligence","Research","Yes","","","","","","","","Yes (Compressed Sensing)","","","","","General FL","Communication-Efficient Federated Learning Based on Compressed Sensing and Ternary Quantization"
"yang2022federated","2022","IEEE Trans. on Parallel and Distributed Systems","Research","","","Yes","","","","","","","","","Yes (Nesterov)","","Yes","General FL","Federated Learning with Nesterov Accelerated Gradient"
"robbins1951sgd","1951","The Annals of Mathematical Statistics","Research (Foundational)","","","","","","","","","","","","","","Yes","Stochastic Approximation (Origin)","A Stochastic Approximation Method"
"farhadkhani2022byzantine","2022","ICML","Research","","Yes","","","","","","","","Yes","Yes (Resilient Averaging)","","Yes","General ML","Byzantine Machine Learning Made Easy by Resilient Averaging of Momentums"
"elmhamdi2021distributed","2021","ICLR","Research","","Yes","Yes","","","","","","","Yes","Yes (Distributed Momentum)","","Yes","General Distributed SGD","Distributed Momentum for Byzantine-Resilient Stochastic Gradient Descent"
"Zhu_2023","2023","IEEE Trans. on Signal and Info. Processing","Research","Yes","Yes","","","","","","","Yes","","Yes","","","General Distributed Learning","Byzantine-Robust Distributed Learning With Compression"
"NEURIPS2018_3328bdf9","2018","NeurIPS","Research","Yes","","","","","","","","","Yes (Sparsification)","","","","Yes","General Distributed Optimization","Gradient Sparsification for Communication-Efficient Distributed Optimization"
"xu2025nesterovacceleratedrobustfederatedlearning","2025","arXiv","Research","","Yes","Yes","","","","","","","Yes","Yes (Nesterov)","","","General FL","Nesterov-Accelerated Robust Federated Learning Over Byzantine Adversaries"
"""

# --- 处理逻辑 ---

# 1. 将字符串形式的 CSV 数据读取为 Pandas DataFrame
# 使用 io.StringIO 将字符串包装成文件对象，以便 pandas 读取
df = pd.read_csv(io.StringIO(csv_data.strip()))

# 2. 定义输出文件名
output_filename = 'FL_42_References_Comparison.xlsx'

# 3. 使用 ExcelWriter 初始化，指定引擎为 openpyxl
writer = pd.ExcelWriter(output_filename, engine='openpyxl')

# 4. 将 DataFrame 写入 Excel 的一个 Sheet 中，不包含行索引
df.to_excel(writer, sheet_name='Comparison', index=False)

# 5. 获取工作簿和工作表对象，以便进行格式设置
workbook  = writer.book
worksheet = writer.sheets['Comparison']

# 6. 简单的格式设置：将第一行（表头）加粗
header_font = Font(bold=True)
for cell in worksheet["1:1"]:
    cell.font = header_font

# 7. 保存 Excel 文件
writer.close()

print(f"成功生成 Excel 文件: {output_filename}")
print("请在当前目录下查看。")
