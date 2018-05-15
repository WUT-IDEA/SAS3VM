
//package fire2;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.List;
import java.util.concurrent.Callable;

public class S3VM implements Callable<double[]> {
	double[] currSolution;
	int dimension;
	String[] file_name_four;
	double ratio;
	List<Integer> test_labels;
	int ncpus;

	public S3VM(double[] currSolution, int dimension, String[] file_name_four, double ratio, List<Integer> test_labels,
			int ncpus) {
		this.currSolution = currSolution;
		this.dimension = dimension;
		this.file_name_four = file_name_four;
		this.ratio = ratio;
		this.test_labels = test_labels;
		this.ncpus = ncpus;
	}

	public static double[] CSA_S3VM(double[] currSolution, int dimension, String[] file_name_four, double ratio,
			List<Integer> test_labels, int ncpus) {
		double[] result = new double[4];
		test(currSolution, file_name_four, ratio, ncpus);
		// System.out.println("CSA end ");
		result = evaluate(test_labels, file_name_four[2], ncpus);
		// System.out.println("eva end ");
		return result;
	}

	private static double[] evaluate(List<Integer> test_labels, String outputs, int k_t) {
		int TP = 0;
		int TN = 0;
		int FP = 0;
		int FN = 0;
		double eva[] = new double[4];
		File out_labels_path = new File(new File(outputs).getName() + "." + k_t + "outputs");
		try {
			int tt_la = 0;
			BufferedReader bu = new BufferedReader(new InputStreamReader(new FileInputStream(out_labels_path)));
			for (String line = bu.readLine(); line != null; line = bu.readLine()) {
				if (test_labels.get(tt_la) > 0) {
					if (Double.valueOf(line) > 0) {
						TP = TP + 1;
					} else {
						FP = FP + 1;
					}
				} else {
					if (Double.valueOf(line) > 0) {
						FN = FN + 1;
					} else {
						TN = TN + 1;
					}
				}
				tt_la++;
			}
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		} catch (IOException e) {
			e.printStackTrace();
		}

		eva[0] = 1.0 * (TP + TN) / (TP + TN + FP + FN);
		eva[1] = 1.0 * TP / (TP + FP);
		eva[2] = 1.0 * TP / (TP + FN);
		eva[3] = 1.0 - 2.0 * TP / (2.0 * TP + FP + FN);
		return eva;
	}

	private static void test(double[] currSolution, String[] file_name_four, double ratio, int ncpus) {
		Runtime runt0 = Runtime.getRuntime();
		String f1 = file_name_four[0];
		String f2 = file_name_four[1];

		String[] comman0 = { "/home/zixy16/codebases/zj/svmlin-v1.03/svmlin", "-A", "3", "-W",
				String.valueOf(currSolution[0]), "-U", String.valueOf(currSolution[1]), "-T", String.valueOf(ncpus),
				"-R", String.valueOf(ratio), f1, f2 };
		Process process1;
		try {
			// System.out.println("test start ");
			process1 = runt0.exec(comman0);
			// process1.getInputStream();
			InputStream stderr1 = process1.getInputStream();
			InputStreamReader isr1 = new InputStreamReader(stderr1, "GBK");
			BufferedReader br11 = new BufferedReader(isr1);
			String line1 = null;
			// System.out.println("test start ");
			while ((line1 = br11.readLine()) != null) {
				// System.out.println(line1);
				// if (line1.contains("DA-SVM took")) {
				// System.out.println(line1);
				//
				// }

			}
			// System.out.println(1);
			process1.waitFor();
		} catch (IOException e2) {

			e2.printStackTrace();
		} catch (InterruptedException e) {

			e.printStackTrace();
		}
		// runt0.e
		Runtime runt = Runtime.getRuntime();
		String t1 = new File(file_name_four[0]).getName() + "."+ncpus+"weights";
		String t2 = file_name_four[2];
		String t3 = file_name_four[3];
		String[] comman = { "/home/zixy16/codebases/zj/svmlin-v1.03/svmlin", "-T", String.valueOf(ncpus), "-f", t1, t2,
				t3 };
		// System.out.println(comman[0] + " " + comman[1] + " " + comman[2] + "
		// " + comman[3] + " " + comman[4]);
		Process process;
		try {
			process = runt.exec(comman);
			InputStream stderr = process.getInputStream();
			InputStreamReader isr = new InputStreamReader(stderr, "GBK");
			BufferedReader br = new BufferedReader(isr);
			String line = null;
			while ((line = br.readLine()) != null) {
				// System.out.println(line);
			}
			int exitVal = process.waitFor();

			// int exitVal = proc.waitFor();
			System.out.println("Process exitValue: " + exitVal);
		} catch (IOException e1) {

			e1.printStackTrace();
		} catch (InterruptedException e) {

			e.printStackTrace();
		}

	}

	@Override
	public double[] call() throws Exception {
		return CSA_S3VM(currSolution, dimension, file_name_four, ratio, test_labels, ncpus);
		// return null;
	}

}
