//package fire2;

import java.util.Arrays;
import java.util.HashSet;
import java.util.Random;
import java.util.Set;

public class RandomX_Y {
	public static void main(String[] args) {
		int array[][] = getRandomX(10, 103);
		for (int i = 0; i < array.length; i++) {
			for (int a : array[i]) {
				System.out.print(a + " ");
			}
			System.out.println();
		}
		for (int kt = 0; kt < 10; ++kt) {
			Arrays.sort(array[kt]);
		}
		for (int i = 0; i < 103; ++i) {
			if (Arrays.binarySearch(array[0], i)>=0) {
				System.out.println(i+"true");
			}
		}

	}

	public static int[][] getRandomX(int X, int Y) {
		int array[][] = new int[X][];// X为x组，y为范围
		int re = Y % X;
		for (int i = 0; i < X; ++i) {
			if (i < re) {
				array[i] = new int[Y / X + 1];
			} else {
				array[i] = new int[Y / X];
			}
		}
		Set<Integer> set = new HashSet<Integer>();

		Set<Integer>[] sett = new HashSet[X];
		for (int i = 0; i < sett.length; ++i) {
			sett[i] = new HashSet<>();
		}
		Random random = new Random();
		int num;
		int kk = 0;
		while (true) {
			int temp0 = set.size();
			num = random.nextInt(Y);
			set.add(num);

			int temp1 = set.size();
			if (temp0 != temp1) {
				// sae=
				sett[kk % X].add(num);
				kk++;
			}
			if (set.size() >= Y) {
				break;
			}
		}

		for (int i = 0; i < X; i++) {
			int q = 0;
			for (Integer a : sett[i]) {
				array[i][q] = a;
				// System.out.println(array[i]);
				q++;
			}

		}

		return array;
	}
}
