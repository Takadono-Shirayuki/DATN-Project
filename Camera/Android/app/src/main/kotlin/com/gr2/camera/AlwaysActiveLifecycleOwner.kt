package com.gr2.camera

import androidx.lifecycle.Lifecycle
import androidx.lifecycle.LifecycleOwner
import androidx.lifecycle.LifecycleRegistry

/**
 * A LifecycleOwner that stays at RESUMED state until [destroy] is explicitly called.
 * Use this to bind CameraX use cases so they keep running when the Activity goes to background,
 * instead of ProcessLifecycleOwner which fires ON_STOP when all activities are stopped.
 *
 * Must be created and destroyed on the main thread.
 */
class AlwaysActiveLifecycleOwner : LifecycleOwner {
    private val registry = LifecycleRegistry(this)

    init {
        registry.currentState = Lifecycle.State.RESUMED
    }

    override val lifecycle: Lifecycle get() = registry

    fun destroy() {
        registry.currentState = Lifecycle.State.DESTROYED
    }
}
