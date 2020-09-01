package com.example.facialeditapp

import android.graphics.Bitmap

class Param {
    var uri: String? = null
    var bmp: Bitmap? = null
    constructor(uri: String?, bmp: Bitmap?) {
        this.uri = uri
        this.bmp = bmp
    }
}