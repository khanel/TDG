package com.example.sbstdemo;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

public class ManualSmokeTest {

    @Test
    public void smoke() {
        assertDoesNotThrow(() -> {
            BaseLogic b = new BaseLogic();
            assertTrue(b.baseNeg() <= 0);
            ChildLogic c = new ChildLogic();
            assertTrue(c.childUsesParentCore() <= 3);
            GrandChildLogic g = new GrandChildLogic();
            assertTrue(g.grandSwitch() >= 300);
        });
    }
}
