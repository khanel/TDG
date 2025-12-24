package com.example.sbstdemo;

public class GrandChildLogic extends ChildLogic {

    public int grandSwitch() {
        int v = 0;
        switch (v) {
            case 0:
                return 300;
            case 1:
                return 301;
            default:
                return 302;
        }
    }

    public int grandIfChain() {
        int x = -5;
        if (x < -10) {
            return 400;
        } else if (x < 0) {
            return 401;
        } else {
            return 402;
        }
    }
}
