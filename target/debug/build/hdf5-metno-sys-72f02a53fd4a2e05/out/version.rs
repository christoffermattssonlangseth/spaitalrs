
    #[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Version {
        pub major: u8,
        pub minor: u8,
        pub micro: u8,
    }

    /// HDF5 library version used at link time (1.14.6)
    pub const HDF5_VERSION: Version = Version { major: 1, minor: 14, micro: 6 };
    