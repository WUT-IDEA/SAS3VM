//package fire2;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.text.SimpleDateFormat;
import java.util.ArrayList;
import java.util.Date;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;

public class Main {
	public final static double PI = 3.141592653589793;

	public static void main(String[] args) throws IOException {

		int k = 10;// 1对折数.0
		int total = 4255;
		String path = "/home/zixy16/codebases/zj/svmlin-v1.03/example/data/t/cre73.txt"; // 数据文件地址

		String[] fun = { "MuSA", "CSA_BA", "CSA_M", "CSA_MwVARCONTROL" };
		String fire = "CSA_MwVARCONTROL";// 方法名
		double ratio = 0.674;// 数据正负比例
		int maxNumFuncEval = 4000;// 最大运行次数
		int stepsPerTemperature = 20;// 固定温度下的最大运行次数
		double Tac = 1, T = 1;// 接受温度,迭代温度
		int ncpc = 10;// 耦合数

		for (k = 10; k <= 100; k = k + 10) {
			KFlod kl = new KFlod();
			kl.kFlod(k, total, path);// 分为k组
			for (int fun_n = 0; fun_n < 4; fun_n++) {
				fire = fun[fun_n];
				for (int k_t = 0; k_t < k; ++k_t) {
					kl.pretest(k_t, k);
					String train_name = "/home/zixy16/codebases/zj/svmlin-v1.03/example/japsvm_tr" + k_t + "_" + k;// 训练数据
					String train_label = "/home/zixy16/codebases/zj/svmlin-v1.03/example/japsvm_tr_la" + k_t + "_" + k;// 训练标签
					String test_name = "/home/zixy16/codebases/zj/svmlin-v1.03/example/japsvm_te" + k_t + "_" + k;// 测试数据
					String test_label = "/home/zixy16/codebases/zj/svmlin-v1.03/example/japsvm_te_la" + k_t + "_" + k;// 测试标签
					String[] file_name_four = { train_name, train_label, test_name, test_label };

					CSA(fire, ratio, maxNumFuncEval, stepsPerTemperature, file_name_four, Tac, T, ncpc, k);

					Clear(file_name_four,k_t);
				}
			}
			Clear(k);
		}
	}

	private static void Clear(int k) {
		for (int i = 0; i < k; ++i) {
			File ff = new File("/home/zixy16/codebases/zj/svmlin-v1.03/example/data/test" + i);
			if (ff.exists())
				ff.delete();
			ff = new File("/home/zixy16/codebases/zj/svmlin-v1.03/example/data/label" + i);
			if (ff.exists())
				ff.delete();
		}

	}

	private static void Clear(String[] file_name_four,int k_t) {
		for (int i = 0; i < file_name_four.length; ++i) {
			File ff = new File(file_name_four[i]);
			if (ff.exists())
				ff.delete();
		}
		File ff = new File(file_name_four[0]);
		ff = new File(ff.getName() + "."+k_t+"outputs");
		if (ff.exists())
			ff.delete();
		ff = new File(file_name_four[0]);
		ff = new File(ff.getName() + "."+k_t+"weights");
		if (ff.exists())
			ff.delete();
		ff = new File(file_name_four[2]);
		ff = new File(ff.getName() + "."+k_t+"weights");
		if (ff.exists())
			ff.delete();
	}

	private static void CSA(String fire, double ratio, int maxNumFuncEval, int stepsPerTemperature,
			String[] file_name_four, double Tac, double T, int ncpus, int kflod) {
		long startTime = System.currentTimeMillis();
		String t1 = new SimpleDateFormat("yyyyMMddHHmmssSSS").format(new Date());
		System.out.println(t1);
		int i, j = 0, k;
		int dimension = 1;
		double r;
		// double beta;
		double bestCostSoFar = 1;
		double optimum = -1;
		double errorTolerance = 0.0001;
		double targetProbVar = 0.5;
		int stopCriterium = 0;
		double probVar, probMean;

		dimension = 2;
		targetProbVar = 0.99;
		errorTolerance = 0.0001;

		double[] currCosts = new double[ncpus];
		double[][] currSolution = new double[ncpus][dimension];
		double[][] probeSolution = new double[ncpus][dimension];
		double[] bestSolutionSoFar = new double[dimension];
		double[] pblty = new double[ncpus];

		double[] bestFMeasureSoFar = new double[4];
		double[][] proFMeasure = new double[ncpus][4];
		double[][] currFMeasure = new double[ncpus][4];

		double[] currCost = new double[ncpus];
		double[] probeCost = new double[ncpus];
		double gamma;
		double maxCurrCost;
		// currSolution初始化
		for (i = 0; i < ncpus; ++i) {
			currSolution[i][0] = 0.01;
			currSolution[i][1] = 1;
		}
		File test_labels_path = new File(file_name_four[3]);
		List<Integer> test_labels = new ArrayList<Integer>();
		try {
			BufferedReader bu = new BufferedReader(new InputStreamReader(new FileInputStream(test_labels_path)));
			for (String line = bu.readLine(); line != null; line = bu.readLine()) {
				test_labels.add(Integer.parseInt(line));
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		double[] currFMeasure_temp = S3VM.CSA_S3VM(currSolution[0], dimension, file_name_four, ratio, test_labels, 0);
		System.out.println("CSA end ");
		for (i = 0; i < ncpus; ++i) {
			currFMeasure[i] = currFMeasure_temp;
			currCost[i] = currFMeasure[i][3];
		}
		bestCostSoFar = currCost[0];
		maxCurrCost = currCost[0];
		gamma = 1;
		// beta = currCost;

		for (i = 0; stopCriterium != 1 && (i < maxNumFuncEval / (ncpus * stepsPerTemperature)); i++) {
			// Fixed temperature loop
			for (j = 0; stopCriterium != 1 && (j < stepsPerTemperature); j++) {
				// generate solution;
				List<Future<double[]>> list = new ArrayList<Future<double[]>>();
				int taskSize = 10;
				ExecutorService pool = Executors.newFixedThreadPool(taskSize);
				for (int nc = 0; nc < ncpus; ++nc) {

					for (k = 0; k < dimension; k++) {
						r = Math.tan(PI * (Math.random() - 0.5));
						probeSolution[nc][k] = currSolution[nc][k] + r * T;
						probeSolution[nc][k] = probeSolution[nc][k] > 10.0 ? 10.0 : probeSolution[nc][k];
						probeSolution[nc][k] = probeSolution[nc][k] < 0.0 ? 0.001 : probeSolution[nc][k];
					}
					// System.out.println("CSA start ");
					Callable c = new S3VM(probeSolution[nc], dimension, file_name_four, ratio, test_labels, nc);
					list.add(pool.submit(c));

					// proFMeasure[nc] = S3VM.CSA_S3VM(probeSolution[nc],
					// dimension, file_name_four, ratio, test_labels,nc);
					// System.out.println("CSA end ");
					// probeCost[nc] = proFMeasure[nc][3];
				}
				pool.shutdown();
				int thr = 0;
				for (Future<double[]> f : list) {

					try {
						proFMeasure[thr] = f.get();
						probeCost[thr] = proFMeasure[thr][3];
						System.out.println(probeCost[thr]);

					} catch (InterruptedException | ExecutionException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
					thr++;
				}
				for (int nd = 0; nd < 10; nd++) {
					System.out.println(probeCost[nd]);
				}

				// System.out.println(bestCostSoFar);
				// 计算当前状态的最大cost
				maxCurrCost = currCost[0];
				for (k = 1; k < ncpus; k++)
					if (currCost[k] > maxCurrCost)
						maxCurrCost = currCost[k];

				// beta = 0.0;
				for (k = 0; k < ncpus; k++) {
					gamma = 0.0;
					// beta += currCosts[k];
					if (fire.equals("MuSA") || fire.equals("CSA_BA")) {
						gamma += Math.exp(-currCosts[k] / Tac);
					} else if (fire.equals("CSA_M") || fire.equals("CSA_MwVARCONTROL")) {
						gamma += Math.exp((currCosts[k] - maxCurrCost) / Tac);
					} else if (fire.equals("MSA")) {
						// do nothing
					}

					if (currCosts[k] < (optimum + errorTolerance))
						stopCriterium = 1;
				}

				// calculate probabilities
				for (k = 0; k < ncpus; k++) {
					pblty[k] = 0.0;
					if (fire.equals("MuSA")) {
						pblty[k] = Math.exp(-probeCost[k] / Tac) / (Math.exp(-probeCost[k] / Tac) + gamma);
					} else if (fire.equals("CSA_BA")) {
						pblty[k] = 1.0 - Math.exp(-currCosts[k] / Tac) / gamma;
					} else if (fire.equals("CSA_M") || fire.equals("CSA_MwVARCONTROL")) {
						pblty[k] = Math.exp((currCosts[k] - maxCurrCost) / Tac) / gamma;
					} else if (fire.equals("MSA")) {
						pblty[k] = Math.exp((currCosts[k] - probeCost[k]) / Tac);
					}

					System.out.println("probeCost:" + probeCost[k]);

					// test acceptance
					if (probeCost[k] <= currCost[k]) {
						// accept
						// accdown++;
						currCost[k] = probeCost[k];// 更新当前cost

						for (int d_t = 0; d_t < dimension; d_t++)// 更新当前状态
							currSolution[k][d_t] = probeSolution[k][d_t];

						for (int d_t = 0; d_t < 4; d_t++)// 更新当前结果
							currFMeasure[k][d_t] = proFMeasure[k][d_t];

						if (currCost[k] < bestCostSoFar) { // So far better cost
															// found!
							// bestCostSoFarFlag = 1;
							// 跟新最优解
							System.out.println(currCost[k]);
							bestCostSoFar = currCost[k];
							for (int d_t = 0; d_t < dimension; d_t++) {
								bestSolutionSoFar[d_t] = currSolution[k][d_t];
							}
							for (int d_t = 0; d_t < 4; d_t++) {
								bestFMeasureSoFar[d_t] = currFMeasure[k][d_t];
							}
						}
					} else {
						// check probability
						if (pblty[k] >= Math.random()) {
							// accept
							// accup++;
							currCost[k] = probeCost[k];// 更新当前cost

							for (int d_t = 0; d_t < dimension; d_t++)// 更新当前状态
								currSolution[k][d_t] = probeSolution[k][d_t];

							for (int d_t = 0; d_t < 4; d_t++)// 更新当前结果
								currFMeasure[k][d_t] = proFMeasure[k][d_t];
						}
					}
				}
				System.out.println(bestCostSoFar);
			} // end of the fixed temperature loop

			if (fire.equals("CSA_MwVARCONTROL")) {
				probMean = pblty[0];
				for (k = 1; k < ncpus; k++)
					probMean = probMean + pblty[k];
				probMean = probMean / ((double) ncpus);
				probVar = 0.0;
				for (k = 0; k < ncpus; k++)
					probVar = probVar + Math.pow(pblty[k] - probMean, 2.0);
				probVar = probVar / ((double) ncpus);
				probVar = probVar * Math.pow(ncpus, 2.0) / ((double) (ncpus - 1));
				if (probVar >= targetProbVar)
					Tac = Tac + 0.05 * Tac;
				else
					Tac = Tac - 0.05 * Tac;
			} else {
				Tac = Tac * Math.log((double) i + 2) / Math.log((double) i + 3);
			}

			// adjust generation temperature
			T = T * ((double) i + 1) / ((double) i + 2);
			// T = T * log(h+2)/log(h+3);
		} // end of the annealing loop
		long endTime = System.currentTimeMillis();
		long runtime = -startTime + endTime;
		System.out.println(runtime);
		for (k = 0; k < dimension; k++)
			System.out.println(bestSolutionSoFar[k]);
		System.out.println("cost = " + bestCostSoFar);
		for (int d_t = 0; d_t < 4; d_t++)
			System.out.println(bestFMeasureSoFar[d_t]);

		String path_so = "/home/zixy16/codebases/zj/svmlin-v1.03/out2/so/cre73_so_" + fire + kflod;
		try {
			FileOutputStream fileout = new FileOutputStream(path_so, true);
			String temp = bestSolutionSoFar[0] + "\t" + bestSolutionSoFar[1] + "\n";
			byte[] br1 = temp.getBytes();
			fileout.write(br1);
			fileout.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		String path_out = "/home/zixy16/codebases/zj/svmlin-v1.03/out2/cre73_" + fire + kflod;
		try {
			FileOutputStream fileout = new FileOutputStream(path_out, true);
			String temp = bestFMeasureSoFar[0] + "\t" + bestFMeasureSoFar[1] + "\t" + bestFMeasureSoFar[2] + "\t"
					+ bestFMeasureSoFar[3] + "\t" + runtime + "\n";
			byte[] br1 = temp.getBytes();
			fileout.write(br1);
			fileout.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

}
