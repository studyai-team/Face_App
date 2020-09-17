package com.example.facialeditapp

import android.graphics.Bitmap

class Param {
    var uri: String? = null
    var bmp: Bitmap? = null
    var type: Int? = null
    constructor(uri: String?, bmp: Bitmap?, type: Int?) {
        this.uri = uri
        this.bmp = bmp
        this.type = type
    }
}