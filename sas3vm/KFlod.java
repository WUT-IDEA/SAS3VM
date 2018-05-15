//package fire2;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

public class KFlod {
	public static void main(String[] args) throws IOException {
		//new KFlod().kFlod(1, 15757);
	}
	public  void pretest(int i1, int k1) {
		int i = i1;
		int k = k1;
		String path = "/home/zixy16/codebases/zj/svmlin-v1.03/example/data/test" + i;
		String path2 = "/home/zixy16/codebases/zj/svmlin-v1.03/example/data/label" + i;
		File file = new File(path);
		File file2 = new File(path2);
		List<String> list = new ArrayList<String>();// 训练数据
		List<String> lable = new ArrayList<String>();// 训练数据标签
		List<String> list_test = new ArrayList<String>();// 测试数据
		List<String> lable_test = new ArrayList<String>();// 测试数据标签
		try {
			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
			for (String line = br.readLine(); line != null; line = br.readLine()) {
				// System.out.println(line);
				list.add(line + "\n");
			}
			br.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		try {
			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file2)));
			for (String line = br.readLine(); line != null; line = br.readLine()) {
				// System.out.println(line);
				lable.add(line + "\n");
			}
			br.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}

		for (int m = 0; m < k; m++) {
			if (m != i) {
				String patht = "/home/zixy16/codebases/zj/svmlin-v1.03/example/data/test" + m;
				String patht2 = "/home/zixy16/codebases/zj/svmlin-v1.03/example/data/label" + m;
				File filet = new File(patht);
				File filet2 = new File(patht2);

				try {
					BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(filet)));
					for (String line = br.readLine(); line != null; line = br.readLine()) {
						// System.out.println(line);
						list.add(line + "\n");
						list_test.add(line + "\n");
						lable.add("0" + "\n");
					}
					br.close();
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}

				try {
					BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(filet2)));
					for (String line = br.readLine(); line != null; line = br.readLine()) {
						// System.out.println(line);
						lable_test.add(line + "\n");
					}
					br.close();
				} catch (Exception e) {
					// TODO Auto-generated catch block
					e.printStackTrace();
				}

			}
		}

		// 输出到文件
		try {
			FileOutputStream fileout = new FileOutputStream("/home/zixy16/codebases/zj/svmlin-v1.03/example/japsvm_tr" + i + "_" + k);
			FileOutputStream fileout2 = new FileOutputStream(
					"/home/zixy16/codebases/zj/svmlin-v1.03/example/japsvm_tr_la" + i + "_" + k);
			for (int n = 0; n < list.size(); n++) {
				byte[] br1 = list.get(n).getBytes();
				fileout.write(br1);
			}
			for (int n = 0; n < lable.size(); n++) {
				byte[] br1 = lable.get(n).getBytes();
				fileout2.write(br1);
			}
			fileout.close();
			fileout2.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		try {
			FileOutputStream fileout = new FileOutputStream("/home/zixy16/codebases/zj/svmlin-v1.03/example/japsvm_te" + i + "_" + k);
			FileOutputStream fileout2 = new FileOutputStream(
					"/home/zixy16/codebases/zj/svmlin-v1.03/example/japsvm_te_la" + i + "_" + k);
			for (int n = 0; n < list_test.size(); n++) {
				byte[] br1 = list_test.get(n).getBytes();
				fileout.write(br1);
			}
			for (int n = 0; n < lable_test.size(); n++) {
				byte[] br1 = lable_test.get(n).getBytes();
				fileout2.write(br1);
			}
			fileout.close();
			fileout2.close();
		} catch (FileNotFoundException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
	public void kFlod(int k, int total, String path) throws IOException {
		int row = 0;

		int flod[][] = RandomX_Y.getRandomX(k, total);
		for (int kt = 0; kt < k; ++kt) {
			Arrays.sort(flod[kt]);
		}
		
		// String path2 = "/usr/local/svmlin-v1.0/example/data/label.txt"; //
		// 数据文件地址
		File file = new File(path);
		// File file2 = new File(path2);

		Set<Integer> set = new HashSet<Integer>();

		Set<String>[] sett = new HashSet[k];
		for (int i = 0; i < sett.length; ++i) {
			sett[i] = new HashSet<>();
		}
		try {
			BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(file)));
			for (String line = br.readLine(); line != null; line = br.readLine()) {
				for (int i = 0; i < k; ++i) {
					if (Arrays.binarySearch(flod[i], row) >= 0) {
						sett[i].add(line + "\n");

						break;
					}
				}
				row++;
			}
			br.close();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		for (int i = 0; i < k; ++i) {
			String pathout = "/home/zixy16/codebases/zj/svmlin-v1.03/example/data/test" + i;
			String pathout2 = "/home/zixy16/codebases/zj/svmlin-v1.03/example/data/label" + i;

			FileOutputStream fileout = new FileOutputStream(pathout);
			FileOutputStream fileout2 = new FileOutputStream(pathout2);
			for (String tq : sett[i]) {
				String[] temp = tq.split("\t");
				String out2 = temp[0] + "\n";
				byte[] br2 = out2.getBytes();
				fileout2.write(br2);
				String out1 = temp[1];
				for (int j = 2; j < temp.length; ++j) {
					out1 = out1 + "\t" + temp[j];
				}
				// out1=out1+"\n";
				byte[] br1 = out1.getBytes();
				fileout.write(br1);

			}
			fileout.close();
			fileout2.close();
		}

	}



}
