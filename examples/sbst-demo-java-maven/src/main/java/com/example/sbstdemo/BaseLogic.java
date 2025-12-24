package com.example.sbstdemo;

public class BaseLogic {

    protected int core(int x) {
        if (x < 0) {
            return -1;
        } else if (x == 0) {
            return 0;
        } else {
            return 1;
        }
    }

    public int baseCore(int x) {
        return core(x);
    }

    public int baseCompare(int a, int b) {
        if (a < b) {
            return 1;
        }
        if (a == b) {
            return 2;
        }
        return 3;
    }

    public boolean baseStringFlag(String s) {
        if (s == null) {
            return false;
        }
        if (s.length() == 0) {
            return false;
        }
        return s.charAt(0) == 'a';
    }

    public int baseNeg() {
        return core(-7);
    }

    public int baseZero() {
        return core(0);
    }

    public int basePos() {
        return core(7);
    }

    public int baseSwitchA() {
        int v = 1;
        switch (v) {
            case 0:
                return 10;
            case 1:
                return 11;
            default:
                return 12;
        }
    }

    public int baseSwitchB() {
        int v = 2;
        switch (v) {
            case 0:
                return 20;
            case 1:
                return 21;
            default:
                return 22;
        }
    }
}
