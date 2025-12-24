package com.example.sbstdemo;

public class ChildLogic extends BaseLogic {

    public int childIf() {
        int a = 3;
        int b = 5;
        if (a > b) {
            return 100;
        } else {
            return 101;
        }
    }

    public int childCompare(int a, int b) {
        if (a > b) {
            return 110;
        } else if (a == b) {
            return 111;
        }
        return 112;
    }

    public int childTernary() {
        int x = 9;
        return (x % 2 == 0) ? 200 : 201;
    }

    public int childUsesParentCore() {
        return core(-1) + core(0) + core(1);
    }
}
